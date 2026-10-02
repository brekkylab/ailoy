// The commands, by name. Each is called in the company's mode, and returns a table, or a
// `Failure` saying what went wrong. The bridge is `BRIDGE`, not `this`: calling through
// `call` would be a native call, and a script may not wait for a command inside one.

Commands <- {};

RATINGS <- ["none", "appalling", "very poor", "poor", "mediocre", "good", "very good", "excellent", "outstanding"];

function RatingName(r) {
	return (r >= 0 && r < RATINGS.len()) ? RATINGS[r] : "none";
}

function StateName(v) {
	switch (GSVehicle.GetState(v)) {
		case GSVehicle.VS_RUNNING: return "running";
		case GSVehicle.VS_STOPPED: return "stopped";
		case GSVehicle.VS_IN_DEPOT: return "in depot";
		case GSVehicle.VS_AT_STATION: return "at station";
		case GSVehicle.VS_BROKEN: return "broken down";
		case GSVehicle.VS_CRASHED: return "crashed";
	}
	return "?";
}

function TypeName(t) {
	switch (t) {
		case GSVehicle.VT_ROAD: return "road";
		case GSVehicle.VT_AIR: return "air";
		case GSVehicle.VT_RAIL: return "rail";
		case GSVehicle.VT_WATER: return "water";
	}
	return "?";
}

function OwnVehicles(company) {
	local list = GSVehicleList();
	list.Valuate(GSVehicle.GetOwner);
	list.KeepValue(company);
	return list;
}

function OwnStations(company) {
	local list = GSStationList(GSStation.STATION_ANY);
	list.Valuate(GSStation.GetOwner);
	list.KeepValue(company);
	return list;
}

function OwnVehicle(v, company) {
	if (typeof v != "integer" || !GSVehicle.IsValidVehicle(v) || GSVehicle.GetOwner(v) != company) {
		return Failure("no vehicle " + v + " of ours");
	}
	return v;
}

/// The tile a road reaches `where` at: a station's road stop, or what `Place` makes of it.
function RoadPlace(where) {
	if (typeof where == "table" && "station" in where && where.station != null) {
		local s = where.station;
		if (!GSStation.IsValidStation(s)) return Failure("no station " + s);
		foreach (type in [GSStation.STATION_TRUCK_STOP, GSStation.STATION_BUS_STOP]) {
			local tiles = GSTileList_StationType(s, type);
			if (!tiles.IsEmpty()) return tiles.Begin();
		}
		return Failure("the station " + s + " has no road stop");
	}
	return Place(where);
}

function CargoAround(tile, w, h, radius) {
	local accepts = [], produces = [];
	foreach (c, _ in GSCargoList()) {
		if (GSTile.GetCargoAcceptance(tile, c, w, h, radius) >= 8) accepts.append(CargoLabel(c));
		if (GSTile.GetCargoProduction(tile, c, w, h, radius) > 0) produces.append(CargoLabel(c));
	}
	return {accepts = accepts, produces = produces};
}

// --- The company ---------------------------------------------------------------------

Commands.setup <- function(args) {
	local id = Arg(args, "company");
	if (Failed(id)) return id;
	local c = GSCompany.ResolveCompanyID(id);
	if (c == GSCompany.COMPANY_INVALID) return Failure("no company " + id);
	BRIDGE.company = c;
	local mode = GSCompanyMode(c);
	local name = Opt(args, "name", "Ailoy Transport");
	if (GSCompany.GetName(c) != name) GSCompany.SetName(name);
	GSCompany.SetPresidentName(Opt(args, "president", "Ailoy"));
	GSCompany.SetAutoRenewStatus(true);
	return Commands.status({});
};

Commands.status <- function(args) {
	local c = BRIDGE.company;
	local counts = {road = 0, air = 0, rail = 0, water = 0};
	foreach (v, _ in OwnVehicles(c)) counts[TypeName(GSVehicle.GetVehicleType(v))]++;
	local last = GSCompany.CURRENT_QUARTER + 1;
	return {
		date = DateString(GSDate.GetCurrentDate()),
		company = GSCompany.GetName(c),
		money = GSCompany.GetBankBalance(c),
		loan = GSCompany.GetLoanAmount(),
		max_loan = GSCompany.GetMaxLoanAmount(),
		loan_step = GSCompany.GetLoanInterval(),
		value = GSCompany.GetQuarterlyCompanyValue(c, GSCompany.CURRENT_QUARTER),
		this_quarter = {
			income = GSCompany.GetQuarterlyIncome(c, GSCompany.CURRENT_QUARTER),
			expenses = GSCompany.GetQuarterlyExpenses(c, GSCompany.CURRENT_QUARTER),
		},
		last_quarter = {
			income = GSCompany.GetQuarterlyIncome(c, last),
			expenses = GSCompany.GetQuarterlyExpenses(c, last),
			cargo_delivered = GSCompany.GetQuarterlyCargoDelivered(c, last),
		},
		vehicles = counts,
		stations = OwnStations(c).Count(),
		map = [GSMap.GetMapSizeX(), GSMap.GetMapSizeY()],
	};
};

Commands.loan <- function(args) {
	local want = Arg(args, "amount");
	if (Failed(want)) return want;
	local step = GSCompany.GetLoanInterval();
	if (want == "max") want = GSCompany.GetMaxLoanAmount();
	else if (want == "min") want = 0;
	else if (typeof want != "integer") return Failure("the loan is an amount, max or min");
	want = min(GSCompany.GetMaxLoanAmount(), max(0, (want + step - 1) / step * step));
	if (want != GSCompany.GetLoanAmount() && !GSCompany.SetLoanAmount(want)) return Fail("setting the loan to " + want);
	return {loan = GSCompany.GetLoanAmount(), money = GSCompany.GetBankBalance(BRIDGE.company)};
};

// --- The world -----------------------------------------------------------------------

Commands.cargos <- function(args) {
	local items = [];
	foreach (c, _ in GSCargoList()) {
		items.append({
			id = c,
			label = CargoLabel(c),
			name = GSCargo.GetName(c),
			freight = GSCargo.IsFreight(c),
			// What one unit pays, carried 50 tiles in 20 days: to compare cargos by.
			pay_50_tiles_20_days = GSCargo.GetCargoIncome(c, 50, 20),
		});
	}
	return {items = items};
};

Commands.towns <- function(args) {
	local near = OptPlace(args, "near");
	if (Failed(near)) return near;
	local list = GSTownList();
	if (near != null) {
		list.Valuate(GSTown.GetDistanceManhattanToTile, near);
		list.Sort(GSList.SORT_BY_VALUE, true);
	} else {
		list.Valuate(GSTown.GetPopulation);
		list.Sort(GSList.SORT_BY_VALUE, false);
	}
	list.KeepTop(Opt(args, "limit", 15));
	local pass = Passengers();
	local items = [];
	foreach (t, _ in list) {
		local item = {
			id = t,
			name = GSTown.GetName(t),
			population = GSTown.GetPopulation(t),
			tile = XY(GSTown.GetLocation(t)),
			passengers_last_month = GSTown.GetLastMonthProduction(t, pass),
			transported_pct = GSTown.GetLastMonthTransportedPercentage(t, pass),
			our_rating = RatingName(GSTown.GetRating(t, BRIDGE.company)),
		};
		if (near != null) item.distance <- GSTown.GetDistanceManhattanToTile(t, near);
		items.append(item);
	}
	return {items = items};
};

function IndustryItem(i) {
	local type = GSIndustry.GetIndustryType(i);
	local produces = [];
	foreach (c, _ in GSIndustryType.GetProducedCargo(type)) {
		produces.append({
			cargo = CargoLabel(c),
			last_month = GSIndustry.GetLastMonthProduction(i, c),
			transported_pct = GSIndustry.GetLastMonthTransportedPercentage(i, c),
		});
	}
	local accepts = [];
	foreach (c, _ in GSIndustryType.GetAcceptedCargo(type)) accepts.append(CargoLabel(c));
	return {
		id = i,
		name = GSIndustry.GetName(i),
		tile = XY(GSIndustry.GetLocation(i)),
		produces = produces,
		accepts = accepts,
		stations_around = GSIndustry.GetAmountOfStationsAround(i),
	};
}

Commands.industries <- function(args) {
	local cargo = OptCargo(args, "cargo");
	if (Failed(cargo)) return cargo;
	local near = OptPlace(args, "near");
	if (Failed(near)) return near;
	local list;
	if (cargo != null) {
		list = Opt(args, "accepting", false) ? GSIndustryList_CargoAccepting(cargo) : GSIndustryList_CargoProducing(cargo);
	} else {
		list = GSIndustryList();
	}
	if (near != null) {
		list.Valuate(GSIndustry.GetDistanceManhattanToTile, near);
		list.Sort(GSList.SORT_BY_VALUE, true);
	}
	list.KeepTop(Opt(args, "limit", 15));
	local items = [];
	foreach (i, _ in list) {
		local item = IndustryItem(i);
		if (near != null) item.distance <- GSIndustry.GetDistanceManhattanToTile(i, near);
		items.append(item);
	}
	return {items = items};
};

/// One character for what is on `tile`.
function TileChar(tile, company) {
	if (GSTile.IsStationTile(tile)) return GSTile.GetOwner(tile) == company ? "S" : "s";
	if (GSRoad.IsRoadDepotTile(tile)) return GSTile.GetOwner(tile) == company ? "D" : "d";
	if (GSBridge.IsBridgeTile(tile)) return "B";
	if (GSTunnel.IsTunnelTile(tile)) return "T";
	if (GSRoad.IsRoadTile(tile)) return "#";
	if (GSRail.IsRailTile(tile)) return "=";
	if (GSTile.IsWaterTile(tile)) return "~";
	if (GSIndustry.IsValidIndustry(GSIndustry.GetIndustryID(tile))) return "I";
	if (GSTile.IsBuildable(tile)) return GSTile.GetSlope(tile) == GSTile.SLOPE_FLAT ? "." : ",";
	local town = GSTile.GetClosestTown(tile);
	if (GSTown.IsValidTown(town) && GSTown.IsWithinTownInfluence(town, tile)) return "h";
	return "x";
}

Commands.map <- function(args) {
	local center = Place(Opt(args, "at", null));
	if (Failed(center)) return center;
	local r = min(Opt(args, "radius", 12), 25);
	local cx = GSMap.GetTileX(center), cy = GSMap.GetTileY(center);
	local x0 = max(1, cx - r), y0 = max(1, cy - r);
	local x1 = min(GSMap.GetMapSizeX() - 2, cx + r), y1 = min(GSMap.GetMapSizeY() - 2, cy + r);
	local rows = [];
	for (local y = y0; y <= y1; y++) {
		local row = "";
		for (local x = x0; x <= x1; x++) row += TileChar(GSMap.GetTileIndex(x, y), BRIDGE.company);
		rows.append(row);
	}
	local towns = [], industries = [], stations = [];
	foreach (t, _ in GSTownList()) {
		local l = GSTown.GetLocation(t);
		local x = GSMap.GetTileX(l), y = GSMap.GetTileY(l);
		if (x >= x0 && x <= x1 && y >= y0 && y <= y1) towns.append({id = t, name = GSTown.GetName(t), tile = XY(l)});
	}
	foreach (i, _ in GSIndustryList()) {
		local l = GSIndustry.GetLocation(i);
		local x = GSMap.GetTileX(l), y = GSMap.GetTileY(l);
		if (x + 4 >= x0 && x <= x1 && y + 4 >= y0 && y <= y1) {
			industries.append({id = i, name = GSIndustry.GetName(i), tile = XY(l)});
		}
	}
	foreach (s, _ in OwnStations(BRIDGE.company)) {
		local l = GSStation.GetLocation(s);
		local x = GSMap.GetTileX(l), y = GSMap.GetTileY(l);
		if (x >= x0 && x <= x1 && y >= y0 && y <= y1) stations.append(StationSummary(s));
	}
	return {
		top_left = [x0, y0],
		legend = ". flat land  , sloped land  # road  = rail  ~ water  h town building  I industry  S our station  s another's station  D our depot  d another's depot  B bridge  T tunnel  x other",
		towns = towns,
		industries = industries,
		our_stations = stations,
		items = rows,
	};
};

Commands.tile <- function(args) {
	local tile = Place(args);
	if (Failed(tile)) return tile;
	local out = {
		tile = XY(tile),
		what = TileChar(tile, BRIDGE.company),
		height = GSTile.GetMaxHeight(tile),
		flat = GSTile.GetSlope(tile) == GSTile.SLOPE_FLAT,
		buildable = GSTile.IsBuildable(tile),
	};
	local town = GSTile.GetClosestTown(tile);
	if (GSTown.IsValidTown(town)) {
		out.closest_town <- {id = town, name = GSTown.GetName(town), in_town = GSTown.IsWithinTownInfluence(town, tile)};
	}
	local ind = GSIndustry.GetIndustryID(tile);
	if (GSIndustry.IsValidIndustry(ind)) out.industry <- {id = ind, name = GSIndustry.GetName(ind)};
	if (GSTile.IsStationTile(tile)) out.station <- StationSummary(GSStation.GetStationID(tile));
	local cargo = CargoAround(tile, 1, 1, 3);
	out.a_stop_here_would_accept <- cargo.accepts;
	out.a_stop_here_would_supply <- cargo.produces;
	return out;
};

Commands.look <- function(args) {
	local tile = Place(args);
	if (Failed(tile)) return tile;
	return {tile = XY(tile), look_at = tile};
};

// --- Building ------------------------------------------------------------------------

function IsPlainRoad(tile) {
	return GSRoad.IsRoadTile(tile) && !GSRoad.IsRoadStationTile(tile) && !GSRoad.IsRoadDepotTile(tile)
		&& !GSBridge.IsBridgeTile(tile) && !GSTunnel.IsTunnelTile(tile);
}

/// How a road stop could go on `tile`: `{how, front}`, or null.
function StopPlan(tile) {
	local sx = GSMap.GetMapSizeX();
	if (GSRoad.IsRoadTile(tile)) {
		if (!IsPlainRoad(tile)) return null;
		local along_x = GSRoad.AreRoadTilesConnected(tile, tile + 1) || GSRoad.AreRoadTilesConnected(tile, tile - 1);
		local along_y = GSRoad.AreRoadTilesConnected(tile, tile + sx) || GSRoad.AreRoadTilesConnected(tile, tile - sx);
		if (along_x && !along_y) return {how = "drive-through", front = tile + 1};
		if (along_y && !along_x) return {how = "drive-through", front = tile + sx};
		return null;
	}
	if (!GSTile.IsBuildable(tile)) return null;
	// Facing a road if there is one, and otherwise any bare tile a road can come in from.
	local bare = null;
	foreach (off in Offsets()) {
		local front = tile + off;
		if (!Inside(front)) continue;
		if (IsPlainRoad(front)) return {how = "beside road", front = front};
		if (bare == null && GSTile.IsBuildable(front)) bare = front;
	}
	if (bare != null) return {how = "new", front = bare};
	return null;
}

function BuildStop(tile, plan, veh_type) {
	if (plan.how == "drive-through") {
		return GSRoad.BuildDriveThroughRoadStation(tile, plan.front, veh_type, GSStation.STATION_JOIN_ADJACENT);
	}
	if (!GSRoad.BuildRoadStation(tile, plan.front, veh_type, GSStation.STATION_JOIN_ADJACENT)) return false;
	if (!BuildRoadRetrying(tile, plan.front)) {
		GSTile.DemolishTile(tile);
		return false;
	}
	return true;
}

Commands.build_station <- function(args) {
	local acc = GSAccounting();
	local cargo = OptCargo(args, "cargo");
	if (Failed(cargo)) return cargo;
	local kind = Opt(args, "kind", (cargo == null || GSCargo.HasCargoClass(cargo, GSCargo.CC_PASSENGERS)) ? "bus" : "truck");
	local veh_type = kind == "bus" ? GSRoad.ROADVEHTYPE_BUS : GSRoad.ROADVEHTYPE_TRUCK;
	local radius = GSStation.GetCoverageRadius(kind == "bus" ? GSStation.STATION_BUS_STOP : GSStation.STATION_TRUCK_STOP);

	local candidates = [];
	if ("tile" in args && args.tile != null) {
		local tile = Tile(args.tile);
		if (Failed(tile)) return tile;
		local plan = StopPlan(tile);
		if (plan == null) return Failure("a road stop cannot go on " + Where(tile) + ": it needs a straight road without junctions, or bare land");
		if ("front" in args && args.front != null) {
			local front = Tile(args.front);
			if (Failed(front)) return front;
			plan.front = front;
		}
		candidates.append({tile = tile, plan = plan});
	} else {
		if (cargo == null) return Failure("say which cargo, to find a place for the stop");
		local mode = Opt(args, "mode", "pickup");
		local near = Opt(args, "near", null);
		local center = Place(near);
		if (Failed(center)) return center;
		local only = ("industry" in near && near.industry != null) ? near.industry : null;
		local reach = null;
		if (only != null) {
			reach = mode == "dropoff" ? GSTileList_IndustryAccepting(only, radius) : GSTileList_IndustryProducing(only, radius);
		}
		foreach (tile, dist in TilesAround(center, Opt(args, "search", 8))) {
			// For an industry, only where it is in reach, not merely another one.
			if (reach != null && !reach.HasItem(tile)) continue;
			local p = GSTile.GetCargoProduction(tile, cargo, 1, 1, radius);
			local a = GSTile.GetCargoAcceptance(tile, cargo, 1, 1, radius);
			local score;
			if (mode == "pickup") { if (p == 0) continue; score = p; }
			else if (mode == "dropoff") { if (a < 8) continue; score = a; }
			else { if (p == 0 || a < 8) continue; score = p + a; }
			local plan = StopPlan(tile);
			if (plan == null) continue;
			local bonus = plan.how == "drive-through" ? 12 : (plan.how == "beside road" ? 8 : 0);
			candidates.append({tile = tile, plan = plan, rank = score * 4 + bonus - dist});
		}
		candidates.sort(ByRank);
		if (candidates.len() == 0) {
			return Failure("no place for a " + kind + " stop that would " + mode + " " + CargoLabel(cargo) + " near there; try a larger --search");
		}
	}
	local tried = [];
	foreach (i, cand in candidates) {
		if (i >= 15) break;
		if (BuildStop(cand.tile, cand.plan, veh_type)) {
			local s = GSStation.GetStationID(cand.tile);
			local here = CargoAround(cand.tile, 1, 1, radius);
			return {
				station = s,
				name = GSStation.GetName(s),
				tile = XY(cand.tile),
				how = cand.plan.how,
				road_enters_at = XYs(RoadEntries(cand.tile)),
				accepts = here.accepts,
				supplies = here.produces,
				cost = acc.GetCosts(),
				look_at = cand.tile,
			};
		}
		tried.append(Where(cand.tile) + " (" + GSError.GetLastErrorString() + ")");
	}
	return Failure("could not build the stop; tried " + tried.len() + " places, the first " + tried[0]);
};

Commands.build_road <- function(args) {
	local acc = GSAccounting();
	local from = RoadPlace(Opt(args, "from", null));
	if (Failed(from)) return from;
	local to = RoadPlace(Opt(args, "to", null));
	if (Failed(to)) return to;
	local done = ConnectRoad(from, to, Opt(args, "max_nodes", 60000));
	if (Failed(done)) return done;
	done.cost <- acc.GetCosts();
	done.look_at <- to;
	return done;
};

Commands.build_depot <- function(args) {
	local acc = GSAccounting();
	local center = RoadPlace(Opt(args, "near", null));
	if (Failed(center)) return center;
	foreach (tile, _ in TilesAround(center, Opt(args, "search", 6))) {
		if (!GSTile.IsBuildable(tile) || GSTile.GetSlope(tile) != GSTile.SLOPE_FLAT) continue;
		foreach (off in Offsets()) {
			local front = tile + off;
			if (!IsPlainRoad(front)) continue;
			if (!GSRoad.BuildRoadDepot(tile, front)) continue;
			if (!BuildRoadRetrying(tile, front)) {
				GSTile.DemolishTile(tile);
				continue;
			}
			return {depot = XY(tile), front = XY(front), cost = acc.GetCosts(), look_at = tile};
		}
	}
	return Fail("no place for a depot beside a road near there");
};

AIRPORTS <- {
	small = GSAirport.AT_SMALL,
	large = GSAirport.AT_LARGE,
	commuter = GSAirport.AT_COMMUTER,
	metropolitan = GSAirport.AT_METROPOLITAN,
	international = GSAirport.AT_INTERNATIONAL,
	intercontinental = GSAirport.AT_INTERCON,
	heliport = GSAirport.AT_HELIPORT,
	helistation = GSAirport.AT_HELISTATION,
	helidepot = GSAirport.AT_HELIDEPOT,
};

function CanBuildAirport(tile, type) {
	local test = GSTestMode();
	local acc = GSAccounting();
	return GSAirport.BuildAirport(tile, type, GSStation.STATION_NEW);
}

Commands.airports <- function(args) {
	local items = [];
	foreach (name, type in AIRPORTS) {
		items.append({
			type = name,
			available = GSAirport.IsValidAirportType(type),
			size = [GSAirport.GetAirportWidth(type), GSAirport.GetAirportHeight(type)],
			catchment = GSAirport.GetAirportCoverageRadius(type),
			price = GSAirport.GetPrice(type),
		});
	}
	return {items = items};
};

Commands.build_airport <- function(args) {
	local acc = GSAccounting();
	local name = Opt(args, "type", "small");
	if (!(name in AIRPORTS)) return Failure("no airport type " + name + "; `airports` lists them");
	local type = AIRPORTS[name];
	if (!GSAirport.IsValidAirportType(type)) return Failure("the " + name + " airport is not available yet");
	local w = GSAirport.GetAirportWidth(type), h = GSAirport.GetAirportHeight(type);
	local radius = GSAirport.GetAirportCoverageRadius(type);
	local pass = Passengers();

	local spots = [];
	if ("tile" in args && args.tile != null) {
		local tile = Tile(args.tile);
		if (Failed(tile)) return tile;
		spots.append({tile = tile, rank = 0});
	} else {
		local center = Place(Opt(args, "near", null));
		if (Failed(center)) return center;
		foreach (tile, dist in TilesAround(center, Opt(args, "search", 12))) {
			if (!GSTile.IsBuildableRectangle(tile, w, h)) continue;
			local a = GSTile.GetCargoAcceptance(tile, pass, w, h, radius);
			if (a < 8) continue;
			spots.append({tile = tile, rank = a * 2 - dist});
		}
		spots.sort(ByRank);
	}
	local chosen = null;
	// Where it goes as the land is, first.
	foreach (i, spot in spots) {
		if (i >= 80) break;
		if (CanBuildAirport(spot.tile, type) && GSAirport.BuildAirport(spot.tile, type, GSStation.STATION_NEW)) {
			chosen = spot.tile;
			break;
		}
	}
	// Then levelling the land for it, on the best few.
	if (chosen == null && Opt(args, "level", true)) {
		foreach (i, spot in spots) {
			if (i >= 4) break;
			if (!GSTile.LevelTiles(spot.tile, spot.tile + GSMap.GetTileIndex(w, h))) continue;
			if (GSAirport.BuildAirport(spot.tile, type, GSStation.STATION_NEW)) {
				chosen = spot.tile;
				break;
			}
		}
	}
	if (chosen == null) {
		if (spots.len() == 0) return Failure("no clear " + w + "x" + h + " land near there in reach of passengers; try a larger --search");
		return Fail("no place for a " + name + " airport near there");
	}
	local s = GSStation.GetStationID(chosen);
	return {
		station = s,
		name = GSStation.GetName(s),
		tile = XY(chosen),
		hangar = XY(GSAirport.GetHangarOfAirport(chosen)),
		passengers_acceptance = GSTile.GetCargoAcceptance(chosen, pass, w, h, radius),
		cost = acc.GetCosts(),
		look_at = chosen,
	};
};

Commands.demolish <- function(args) {
	local acc = GSAccounting();
	local tile = Tile(Opt(args, "tile", null));
	if (Failed(tile)) return tile;
	if (!GSTile.DemolishTile(tile)) return Fail("demolishing " + Where(tile));
	return {cost = acc.GetCosts()};
};

// --- Vehicles ------------------------------------------------------------------------

function PlaneTypeName(e) {
	switch (GSEngine.GetPlaneType(e)) {
		case GSAirport.PT_HELICOPTER: return "helicopter";
		case GSAirport.PT_SMALL_PLANE: return "small";
		case GSAirport.PT_BIG_PLANE: return "big (crashes often on small airports)";
	}
	return "?";
}

Commands.engines <- function(args) {
	local type_name = Opt(args, "type", "road");
	local type = type_name == "air" ? GSVehicle.VT_AIR : GSVehicle.VT_ROAD;
	local cargo = OptCargo(args, "cargo");
	if (Failed(cargo)) return cargo;
	local list = GSEngineList(type);
	list.Valuate(GSEngine.IsBuildable);
	list.KeepValue(1);
	local items = [];
	foreach (e, _ in list) {
		if (type == GSVehicle.VT_ROAD && GSEngine.GetRoadType(e) != GSRoad.ROADTYPE_ROAD) continue;
		if (cargo != null && GSEngine.GetCargoType(e) != cargo && !GSEngine.CanRefitCargo(e, cargo)) continue;
		local item = {
			id = e,
			name = GSEngine.GetName(e),
			cargo = CargoLabel(GSEngine.GetCargoType(e)),
			capacity = GSEngine.GetCapacity(e),
			speed_kmh = GSEngine.GetMaxSpeed(e),
			price = GSEngine.GetPrice(e),
			running_cost_per_year = GSEngine.GetRunningCost(e),
			reliability_pct = GSEngine.GetReliability(e),
			lifespan_years = GSEngine.GetMaxAge(e) / 366,
		};
		if (cargo != null && GSEngine.GetCargoType(e) != cargo) item.refit_to <- CargoLabel(cargo);
		if (type == GSVehicle.VT_AIR) {
			item.plane <- PlaneTypeName(e);
			local range = GSEngine.GetMaximumOrderDistance(e);
			if (range > 0) item.range_tiles <- range;
		}
		items.append(item);
	}
	return {items = items};
};

function OrderFlags(stop) {
	local flags = GSOrder.OF_NONE;
	foreach (f in Opt(stop, "flags", [])) {
		switch (f) {
			case "full": flags = flags | GSOrder.OF_FULL_LOAD_ANY; break;
			case "full_all": flags = flags | GSOrder.OF_FULL_LOAD; break;
			case "unload": flags = flags | GSOrder.OF_UNLOAD; break;
			case "transfer": flags = flags | GSOrder.OF_TRANSFER; break;
			case "noload": flags = flags | GSOrder.OF_NO_LOAD; break;
			case "nounload": flags = flags | GSOrder.OF_NO_UNLOAD; break;
			default: return Failure("no order flag " + f + "; they are full, full_all, unload, transfer, noload and nounload");
		}
	}
	return flags;
}

function ClearOrders(v) {
	GSOrder.UnshareOrders(v);
	while (GSOrder.GetOrderCount(v) > 0) {
		if (!GSOrder.RemoveOrder(v, 0)) return Fail("clearing the orders of " + v);
	}
	return null;
}

/// Check `stops` before anything is bought or changed for them.
function CheckStops(stops) {
	if (typeof stops != "array" || stops.len() == 0) return Failure("orders are a list of stops");
	foreach (stop in stops) {
		local s = Arg(stop, "station");
		if (Failed(s)) return s;
		if (!GSStation.IsValidStation(s)) return Failure("no station " + s);
		local flags = OrderFlags(stop);
		if (Failed(flags)) return flags;
	}
	return null;
}

function SetOrders(v, stops) {
	local cleared = ClearOrders(v);
	if (Failed(cleared)) return cleared;
	foreach (stop in stops) {
		if (!GSOrder.AppendOrder(v, GSStation.GetLocation(stop.station), OrderFlags(stop))) {
			return Fail("ordering vehicle " + v + " to station " + stop.station);
		}
	}
	return null;
}

function OrdersOf(v) {
	local out = [];
	for (local i = 0; i < GSOrder.GetOrderCount(v); i++) {
		local tile = GSOrder.GetOrderDestination(v, i);
		if (GSTile.IsStationTile(tile)) {
			local flags = GSOrder.GetOrderFlags(v, i);
			local f = [];
			if ((flags & GSOrder.OF_FULL_LOAD_ANY) == GSOrder.OF_FULL_LOAD_ANY) f.append("full");
			else if (flags & GSOrder.OF_FULL_LOAD) f.append("full_all");
			if (flags & GSOrder.OF_UNLOAD) f.append("unload");
			if (flags & GSOrder.OF_TRANSFER) f.append("transfer");
			if (flags & GSOrder.OF_NO_LOAD) f.append("noload");
			if (flags & GSOrder.OF_NO_UNLOAD) f.append("nounload");
			out.append({station = GSStation.GetStationID(tile), flags = f});
		} else {
			out.append({depot = XY(tile)});
		}
	}
	return out;
}

/// What `v` carries most of: an aircraft carries mail besides its passengers.
function VehicleCargo(v) {
	local best = null, most = 0;
	foreach (c, _ in GSCargoList()) {
		local cap = GSVehicle.GetCapacity(v, c);
		if (cap > most) { best = c; most = cap; }
	}
	return best;
}

function VehicleItem(v) {
	local c = VehicleCargo(v);
	return {
		id = v,
		type = TypeName(GSVehicle.GetVehicleType(v)),
		engine = GSEngine.GetName(GSVehicle.GetEngineType(v)),
		state = StateName(v),
		tile = XY(GSVehicle.GetLocation(v)),
		profit_this_year = GSVehicle.GetProfitThisYear(v),
		profit_last_year = GSVehicle.GetProfitLastYear(v),
		age_days = GSVehicle.GetAge(v),
		reliability_pct = GSVehicle.GetReliability(v),
		cargo = c == null ? null : CargoLabel(c),
		load = c == null ? 0 : GSVehicle.GetCargoLoad(v, c),
		capacity = c == null ? 0 : GSVehicle.GetCapacity(v, c),
		orders = OrdersOf(v),
	};
}

/// The company's depot for vehicles of `type` nearest `tile`, or null.
function NearestDepot(company, type, tile) {
	local list = GSDepotList(type == GSVehicle.VT_AIR ? GSTile.TRANSPORT_AIR : GSTile.TRANSPORT_ROAD);
	list.Valuate(GSTile.GetOwner);
	list.KeepValue(company);
	if (list.IsEmpty()) return null;
	list.Valuate(GSMap.DistanceManhattan, tile);
	list.Sort(GSList.SORT_BY_VALUE, true);
	return list.Begin();
}

Commands.buy <- function(args) {
	local acc = GSAccounting();
	local engine = Arg(args, "engine");
	if (Failed(engine)) return engine;
	if (!GSEngine.IsBuildable(engine)) return Failure("the engine " + engine + " is not on sale; `engines` lists those that are");
	local depot = Tile(Opt(args, "depot", null));
	if (Failed(depot)) return depot;
	local cargo = OptCargo(args, "cargo");
	if (Failed(cargo)) return cargo;
	local stops = Opt(args, "orders", null);
	if (stops != null) {
		local bad = CheckStops(stops);
		if (Failed(bad)) return bad;
	}
	local share = Opt(args, "share_with", null);
	if (share != null) {
		share = OwnVehicle(share, BRIDGE.company);
		if (Failed(share)) return share;
	}
	local count = Opt(args, "count", 1);
	local bought = [];
	local note = null;
	for (local n = 0; n < count; n++) {
		local v = GSVehicle.BuildVehicle(depot, engine);
		if (!GSVehicle.IsValidVehicle(v)) {
			if (bought.len() == 0) return Fail("buying at " + Where(depot));
			note = "bought " + bought.len() + " of " + count + ": " + GSError.GetLastErrorString();
			break;
		}
		if (cargo != null && GSVehicle.GetCapacity(v, cargo) == 0 && !GSVehicle.RefitVehicle(v, cargo)) {
			local why = GSError.GetLastErrorString();
			GSVehicle.SellVehicle(v);
			return Failure("refitting to " + CargoLabel(cargo) + " failed (" + why + ")");
		}
		local ordered = null;
		if (share != null) {
			if (!GSOrder.ShareOrders(v, share)) ordered = Fail("sharing the orders of " + share);
		} else if (stops != null) {
			if (bought.len() == 0) ordered = SetOrders(v, stops);
			else if (!GSOrder.ShareOrders(v, bought[0])) ordered = Fail("sharing orders");
		}
		bought.append(v);
		if (Failed(ordered)) {
			note = ordered.message;
			break;
		}
		if (Opt(args, "start", true) && GSOrder.GetOrderCount(v) > 0) GSVehicle.StartStopVehicle(v);
	}
	local out = {vehicles = bought, cost = acc.GetCosts()};
	local c = VehicleCargo(bought[0]);
	out.cargo <- c == null ? null : CargoLabel(c);
	out.capacity <- c == null ? 0 : GSVehicle.GetCapacity(bought[0], c);
	if (note == null && GSOrder.GetOrderCount(bought[0]) == 0) note = "no orders: the vehicles wait in the depot";
	if (note != null) out.note <- note;
	return out;
};

Commands.orders <- function(args) {
	local vehicles = Arg(args, "vehicles");
	if (Failed(vehicles)) return vehicles;
	foreach (v in vehicles) {
		local ok = OwnVehicle(v, BRIDGE.company);
		if (Failed(ok)) return ok;
	}
	local share = Opt(args, "share_with", null);
	local stops = null;
	if (share != null) {
		share = OwnVehicle(share, BRIDGE.company);
		if (Failed(share)) return share;
	} else {
		stops = Opt(args, "orders", null);
		local bad = CheckStops(stops);
		if (Failed(bad)) return bad;
	}
	local out = [];
	foreach (v in vehicles) {
		if (share != null) {
			local cleared = ClearOrders(v);
			if (Failed(cleared)) return cleared;
			if (!GSOrder.ShareOrders(v, share)) return Fail("sharing the orders of " + share);
		} else {
			local set = SetOrders(v, stops);
			if (Failed(set)) return set;
		}
		out.append({id = v, orders = OrdersOf(v)});
	}
	return {items = out};
};

Commands.vehicles <- function(args) {
	local list = OwnVehicles(BRIDGE.company);
	if ("station" in args && args.station != null) {
		if (!GSStation.IsValidStation(args.station)) return Failure("no station " + args.station);
		list.KeepList(GSVehicleList_Station(args.station));
	}
	if ("ids" in args && args.ids != null) {
		local keep = GSList();
		foreach (v in args.ids) keep.AddItem(v, 0);
		list.KeepList(keep);
	}
	local items = [];
	foreach (v, _ in list) items.append(VehicleItem(v));
	return {items = items};
};

Commands.vehicle <- function(args) {
	local action = Opt(args, "action", "");
	if (!Contains(["start", "stop", "depot", "sell", "clone"], action)) {
		return Failure("the actions are start, stop, depot, sell and clone");
	}
	local ids = Arg(args, "ids");
	if (Failed(ids)) return ids;
	foreach (v in ids) {
		local ok = OwnVehicle(v, BRIDGE.company);
		if (Failed(ok)) return ok;
	}
	local out = [];
	foreach (v in ids) {
		local state = GSVehicle.GetState(v);
		local r = {id = v};
		if (action == "start") {
			if (state == GSVehicle.VS_STOPPED || state == GSVehicle.VS_IN_DEPOT) {
				if (!GSVehicle.StartStopVehicle(v)) return Fail("starting " + v);
			}
		} else if (action == "stop") {
			if (state != GSVehicle.VS_STOPPED && state != GSVehicle.VS_IN_DEPOT) {
				if (!GSVehicle.StartStopVehicle(v)) return Fail("stopping " + v);
			}
		} else if (action == "depot") {
			if (!GSVehicle.SendVehicleToDepot(v)) return Fail("sending " + v + " to a depot");
		} else if (action == "sell") {
			if (GSVehicle.IsStoppedInDepot(v)) {
				if (!GSVehicle.SellVehicle(v)) return Fail("selling " + v);
				r.sold <- true;
			} else {
				if (!GSVehicle.SendVehicleToDepot(v)) return Fail("sending " + v + " to a depot");
				BRIDGE.to_sell.append(v);
				r.sold <- "when it reaches a depot";
			}
		} else {
			local type = GSVehicle.GetVehicleType(v);
			local near = GSOrder.GetOrderCount(v) > 0 ? GSOrder.GetOrderDestination(v, 0) : GSVehicle.GetLocation(v);
			local depot;
			if ("depot" in args && args.depot != null) {
				depot = Tile(args.depot);
				if (Failed(depot)) return depot;
			} else {
				depot = NearestDepot(BRIDGE.company, type, near);
				if (depot == null) return Failure("no depot to build the copies at");
			}
			local copies = [];
			for (local n = 0; n < Opt(args, "count", 1); n++) {
				local copy = GSVehicle.CloneVehicle(depot, v, true);
				if (!GSVehicle.IsValidVehicle(copy)) {
					if (copies.len() == 0) return Fail("cloning " + v + " at " + Where(depot));
					r.note <- GSError.GetLastErrorString();
					break;
				}
				GSVehicle.StartStopVehicle(copy);
				copies.append(copy);
			}
			r.copies <- copies;
		}
		out.append(r);
	}
	return {items = out};
};

// --- Stations and how it is going ----------------------------------------------------

function StationKinds(s) {
	local kinds = [];
	if (GSStation.HasStationType(s, GSStation.STATION_BUS_STOP)) kinds.append("bus");
	if (GSStation.HasStationType(s, GSStation.STATION_TRUCK_STOP)) kinds.append("truck");
	if (GSStation.HasStationType(s, GSStation.STATION_AIRPORT)) kinds.append("airport");
	if (GSStation.HasStationType(s, GSStation.STATION_TRAIN)) kinds.append("rail");
	if (GSStation.HasStationType(s, GSStation.STATION_DOCK)) kinds.append("dock");
	return kinds;
}

function StationCargo(s) {
	local out = [];
	foreach (c, _ in GSCargoList()) {
		if (!GSStation.HasCargoRating(s, c)) continue;
		out.append({cargo = CargoLabel(c), waiting = GSStation.GetCargoWaiting(s, c), rating_pct = GSStation.GetCargoRating(s, c)});
	}
	return out;
}

Commands.stations <- function(args) {
	local items = [];
	foreach (s, _ in OwnStations(BRIDGE.company)) {
		local item = StationSummary(s);
		item.kinds <- StationKinds(s);
		item.cargo <- StationCargo(s);
		item.vehicles <- GSVehicleList_Station(s).Count();
		local town = GSStation.GetNearestTown(s);
		if (GSTown.IsValidTown(town)) item.town <- GSTown.GetName(town);
		if (GSStation.HasStationType(s, GSStation.STATION_AIRPORT)) {
			local tile = GSTileList_StationType(s, GSStation.STATION_AIRPORT).Begin();
			item.hangar <- XY(GSAirport.GetHangarOfAirport(tile));
		}
		items.append(item);
	}
	return {items = items};
};

Commands.report <- function(args) {
	local out = Commands.status({});
	local problems = [];
	local profit = 0, profit_last = 0;
	local month = GSDate.GetMonth(GSDate.GetCurrentDate());
	foreach (v, _ in OwnVehicles(BRIDGE.company)) {
		profit += GSVehicle.GetProfitThisYear(v);
		profit_last += GSVehicle.GetProfitLastYear(v);
		local state = GSVehicle.GetState(v);
		local age = GSVehicle.GetAge(v);
		if (state == GSVehicle.VS_CRASHED) {
			problems.append("vehicle " + v + " crashed");
		} else if (state == GSVehicle.VS_STOPPED || state == GSVehicle.VS_IN_DEPOT) {
			if (!Contains(BRIDGE.to_sell, v)) problems.append("vehicle " + v + " is stopped");
		} else if (age > 500 && GSVehicle.GetProfitLastYear(v) < 0) {
			problems.append("vehicle " + v + " lost " + (-GSVehicle.GetProfitLastYear(v)) + " last year");
		} else if (age > 200 && month > 3 && GSVehicle.GetProfitThisYear(v) <= 0 && GSVehicle.GetProfitLastYear(v) <= 0) {
			problems.append("vehicle " + v + " has earned nothing yet");
		}
	}
	foreach (s, _ in OwnStations(BRIDGE.company)) {
		local name = "station " + s + " (" + GSStation.GetName(s) + ")";
		foreach (c in StationCargo(s)) {
			if (c.waiting > 150) problems.append(name + " has " + c.waiting + " " + c.cargo + " waiting");
			if (c.rating_pct < 40) problems.append(name + " rates " + c.rating_pct + "% for " + c.cargo);
		}
		if (GSVehicleList_Station(s).Count() == 0) problems.append(name + " has no vehicles");
	}
	out.vehicle_profit_this_year <- profit;
	out.vehicle_profit_last_year <- profit_last;
	out.problems <- problems;
	return out;
};

