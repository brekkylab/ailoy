// Roads: an A* over tiles to find one, and the commands to build what it found.
//
// A step is one tile over, onto a road, a buildable tile or a railway crossed at right
// angles. Where a step cannot go, a bridge or a tunnel may, from a tile in line with the
// way the road came; a bridge or a tunnel already there is taken as one long step.

const STEP = 0;
const NEW_BRIDGE = 1;
const NEW_TUNNEL = 2;
const OLD_CROSSING = 3;

/// Where a road leaves `tile`: the tiles in front of a station or a depot, or the tile itself.
function RoadEntries(tile) {
	if (GSRoad.IsRoadStationTile(tile)) {
		local front = GSRoad.GetRoadStationFrontTile(tile);
		if (GSRoad.IsDriveThroughRoadStationTile(tile)) return [front, GSRoad.GetDriveThroughBackTile(tile)];
		return [front];
	}
	if (GSRoad.IsRoadDepotTile(tile)) return [GSRoad.GetRoadDepotFrontTile(tile)];
	return [tile];
}

// A test runs in its own accounting: what a test would cost is counted as spent otherwise,
// and a command's cost would take in every test the search made.
function CanBuildBridge(bridge, from, to) {
	local test = GSTestMode();
	local acc = GSAccounting();
	return GSBridge.BuildBridge(GSVehicle.VT_ROAD, bridge, from, to);
}

function CanBuildTunnel(from) {
	local test = GSTestMode();
	local acc = GSAccounting();
	return GSTunnel.BuildTunnel(GSVehicle.VT_ROAD, from);
}

/// A bridge `length` tiles long, head to head, or null: the cheapest that does not slow a
/// road vehicle of the time, and otherwise the fastest.
function BestBridge(length) {
	local list = GSBridgeList_Length(length);
	if (list.IsEmpty()) return null;
	local fastest = GSList();
	fastest.AddList(list);
	fastest.Valuate(GSBridge.GetMaxSpeed);
	fastest.Sort(GSList.SORT_BY_VALUE, false);
	list.Valuate(GSBridge.GetMaxSpeed);
	list.KeepAboveValue(87);
	if (list.IsEmpty()) return fastest.Begin();
	list.Valuate(GSBridge.GetPrice, length);
	list.Sort(GSList.SORT_BY_VALUE, true);
	return list.Begin();
}

/// A railway `next` that a road stepping `off` onto it would cross at right angles.
function IsCrossable(next, off) {
	if (!GSRail.IsRailTile(next) || GSRail.IsRailDepotTile(next) || GSRail.IsRailStationTile(next)) return false;
	local want = (off == 1 || off == -1) ? GSRail.RAILTRACK_NW_SE : GSRail.RAILTRACK_NE_SW;
	return GSRail.GetRailTracks(next) == want;
}

/// What A* guesses is left from `tile`: a little under a new road's cost for each tile.
function Estimate(tile, goals) {
	local best = 1 << 30;
	foreach (g in goals) best = min(best, GSMap.DistanceManhattan(tile, g));
	return best * 5;
}

/// A road from any of `starts` to any of `goals`: the tiles and how each was reached, or null.
function FindRoad(starts, goals, blocked, max_nodes) {
	local goal_set = {};
	foreach (g in goals) goal_set[g] <- true;
	local queue = GSPriorityQueue();
	local came = {};
	local cost = {};
	local closed = {};

	foreach (s in starts) {
		came[s] <- {prev = null, kind = STEP, dir = 0, via = null};
		cost[s] <- 0;
		queue.Insert(s, Estimate(s, goals));
	}

	local expanded = 0;
	while (!queue.IsEmpty()) {
		local cur = queue.Pop();
		if (cur in closed) continue;
		closed[cur] <- true;
		if (cur in goal_set) {
			local path = [];
			local t = cur;
			while (t != null) {
				local c = came[t];
				path.append({tile = t, kind = c.kind, via = c.via});
				t = c.prev;
			}
			path.reverse();
			return path;
		}
		if (++expanded > max_nodes) return null;

		local info = came[cur];
		local prev = info.prev;
		local cur_cost = cost[cur];
		// Off a bridge or out of a tunnel, the road goes straight on.
		local offsets = info.kind != STEP ? [info.dir] : Offsets();

		foreach (off in offsets) {
			local next = cur + off;
			if (next == prev || !Inside(next) || (next in blocked)) continue;
			local came_dir = prev == null ? 0 : (info.kind == STEP ? cur - prev : info.dir);
			local turn = (came_dir != 0 && came_dir != off) ? 2 : 0;

			// Whatever is on `cur` has to take a road in from where it came and out to `next`.
			if (info.kind == STEP && prev != null && !GSRoad.IsRoadStationTile(cur)
					&& GSRoad.CanBuildConnectedRoadPartsHere(cur, prev, next) == 0) continue;

			// A bridge or a tunnel already there, in line: over it in one go.
			if (GSBridge.IsBridgeTile(next) || GSTunnel.IsTunnelTile(next)) {
				if (!GSTile.HasTransportType(next, GSTile.TRANSPORT_ROAD)) continue;
				local other = GSBridge.IsBridgeTile(next) ? GSBridge.GetOtherBridgeEnd(next) : GSTunnel.GetOtherTunnelEnd(next);
				local len = GSMap.DistanceManhattan(next, other);
				if (other != next + off * len || (other in closed)) continue;
				local c = cur_cost + 4 * (len + 1) + turn;
				if (!(other in cost) || c < cost[other]) {
					cost[other] <- c;
					came[other] <- {prev = cur, kind = OLD_CROSSING, dir = off, via = next};
					queue.Insert(other, c + Estimate(other, goals));
				}
				continue;
			}

			local step = null;
			if (next in goal_set) {
				step = 3;
			} else if (GSRoad.IsRoadTile(next)) {
				if (GSRoad.IsRoadStationTile(next) || GSRoad.IsRoadDepotTile(next)) continue;
				step = 3;
			} else if (GSTile.IsBuildable(next)) {
				step = GSTile.GetSlope(next) == GSTile.SLOPE_FLAT ? 10 : 14;
			} else if (IsCrossable(next, off)) {
				step = 25;
			}
			if (step != null) {
				if (next in closed) continue;
				local c = cur_cost + step + turn;
				if (!(next in cost) || c < cost[next]) {
					cost[next] <- c;
					came[next] <- {prev = cur, kind = STEP, dir = off, via = null};
					queue.Insert(next, c + Estimate(next, goals));
				}
				continue;
			}

			// In the way: over it or under it, from a bare tile the road came into in line.
			if (!GSTile.IsBuildable(cur) || (came_dir != 0 && came_dir != off) || info.kind != STEP) continue;
			for (local len = 2; len <= 14; len++) {
				local end = cur + off * len;
				if (!Inside(end)) break;
				if (!GSTile.IsBuildable(end) || (end in closed) || (end in blocked)) continue;
				local bridge = BestBridge(len + 1);
				if (bridge == null) continue;
				if (!CanBuildBridge(bridge, cur, end)) continue;
				local c = cur_cost + 30 * (len + 1) + 40;
				if (!(end in cost) || c < cost[end]) {
					cost[end] <- c;
					came[end] <- {prev = cur, kind = NEW_BRIDGE, dir = off, via = null};
					queue.Insert(end, c + Estimate(end, goals));
				}
				break;
			}
			if (GSTile.GetSlope(cur) != GSTile.SLOPE_FLAT) {
				local end = GSTunnel.GetOtherTunnelEnd(cur);
				if (GSMap.IsValidTile(end) && end != cur) {
					local len = GSMap.DistanceManhattan(cur, end);
					if (end == cur + off * len && len <= 20 && !(end in closed) && CanBuildTunnel(cur)) {
						local c = cur_cost + 25 * (len + 1) + 40;
						if (!(end in cost) || c < cost[end]) {
							cost[end] <- c;
							came[end] <- {prev = cur, kind = NEW_TUNNEL, dir = off, via = null};
							queue.Insert(end, c + Estimate(end, goals));
						}
					}
				}
			}
		}
	}
	return null;
}

/// Build a road from `a` to `b`, waiting out vehicles in the way. True if it is there.
function BuildRoadRetrying(a, b) {
	for (local i = 0; i < 10; i++) {
		if (GSRoad.BuildRoad(a, b)) return true;
		local err = GSError.GetLastError();
		if (err == GSError.ERR_ALREADY_BUILT) return true;
		if (err != GSError.ERR_VEHICLE_IN_THE_WAY) return false;
		GSController.Sleep(10);
	}
	return false;
}

/// Build `path` from `FindRoad`. Null if all of it is built, or the tile where it failed.
function BuildPath(path) {
	// Bridges and tunnels first: a head's tile has to be bare, and the road up to it would
	// not leave it so.
	for (local i = 1; i < path.len(); i++) {
		local a = path[i - 1].tile, b = path[i].tile;
		if (path[i].kind == NEW_BRIDGE) {
			local bridge = BestBridge(GSMap.DistanceManhattan(a, b) + 1);
			if (bridge == null || !GSBridge.BuildBridge(GSVehicle.VT_ROAD, bridge, a, b)) {
				if (GSError.GetLastError() != GSError.ERR_ALREADY_BUILT) return {tile = a, error = GSError.GetLastErrorString()};
			}
		} else if (path[i].kind == NEW_TUNNEL) {
			if (!GSTunnel.BuildTunnel(GSVehicle.VT_ROAD, a)) {
				if (GSError.GetLastError() != GSError.ERR_ALREADY_BUILT) return {tile = a, error = GSError.GetLastErrorString()};
			}
		}
	}
	// Then the road, a straight run at a time.
	local i = 1;
	while (i < path.len()) {
		local kind = path[i].kind;
		if (kind == NEW_BRIDGE || kind == NEW_TUNNEL) { i++; continue; }
		if (kind == OLD_CROSSING) {
			local a = path[i - 1].tile, via = path[i].via;
			if (!BuildRoadRetrying(a, via)) return {tile = via, error = GSError.GetLastErrorString()};
			i++;
			continue;
		}
		local start = path[i - 1].tile;
		local dir = path[i].tile - start;
		local j = i;
		while (j + 1 < path.len() && path[j + 1].kind == STEP && path[j + 1].tile - path[j].tile == dir) j++;
		local end = path[j].tile;
		if (!BuildRoadRetrying(start, end)) {
			if (GSError.GetLastError() == GSError.ERR_NOT_ENOUGH_CASH) return {tile = start, error = "not enough money", fatal = true};
			// Tile by tile, to find the one that fails.
			for (local k = i; k <= j; k++) {
				local a = path[k - 1].tile, b = path[k].tile;
				if (!BuildRoadRetrying(a, b)) return {tile = b, error = GSError.GetLastErrorString()};
			}
		}
		i = j + 1;
	}
	return null;
}

/// Find and build a road from `from` to `to`, going round what fails to build.
function ConnectRoad(from, to, max_nodes) {
	local starts = RoadEntries(from);
	local goals = RoadEntries(to);
	local blocked = {};
	local failures = [];
	for (local attempt = 0; attempt < 4; attempt++) {
		local path = FindRoad(starts, goals, blocked, max_nodes);
		if (path == null) {
			local msg = "no road found from " + Where(from) + " to " + Where(to);
			if (failures.len() > 0) msg += " after building failed at " + failures.len() + " places";
			return Failure(msg);
		}
		local failed = BuildPath(path);
		if (failed == null) {
			local bridges = 0, tunnels = 0;
			foreach (p in path) {
				if (p.kind == NEW_BRIDGE) bridges++;
				if (p.kind == NEW_TUNNEL) tunnels++;
			}
			return {tiles = path.len(), bridges = bridges, tunnels = tunnels, attempts = attempt + 1};
		}
		if ("fatal" in failed) return Failure("stopped building the road at " + Where(failed.tile) + ": " + failed.error);
		failures.append(failed);
		blocked[failed.tile] <- true;
	}
	return Failure("building the road kept failing: last at " + Where(failures.top().tile) + " (" + failures.top().error + ")");
}
