function min(a, b) { return a < b ? a : b; }
function max(a, b) { return a > b ? a : b; }

/// What a command, or a step of one, returns when it cannot go on.
///
/// Not an exception: a script may not wait for a command to be done inside `try`, and
/// OpenTTD ends one that does, so nothing here throws and every caller checks.
class Failure {
	message = null;
	constructor(message) {
		this.message = message;
	}
}

function Failed(x) {
	return typeof x == "instance" && x instanceof Failure;
}

/// A failure saying `what`, and OpenTTD's reason when it has one.
function Fail(what) {
	local err = GSError.GetLastErrorString();
	if (err != null && err != "ERR_NONE") what += " (" + err + ")";
	return Failure(what);
}

/// The tile of an `[x, y]`.
function Tile(xy) {
	if (typeof xy != "array" || xy.len() != 2) return Failure("a tile is [x, y]");
	local x = xy[0], y = xy[1];
	if (x <= 0 || y <= 0 || x >= GSMap.GetMapSizeX() - 1 || y >= GSMap.GetMapSizeY() - 1) {
		return Failure("the tile " + x + "," + y + " is not on the map");
	}
	return GSMap.GetTileIndex(x, y);
}

function XY(tile) {
	return [GSMap.GetTileX(tile), GSMap.GetTileY(tile)];
}

function XYs(tiles) {
	local out = [];
	foreach (t in tiles) out.append(XY(t));
	return out;
}

function Where(tile) {
	return GSMap.GetTileX(tile) + "," + GSMap.GetTileY(tile);
}

/// A required argument.
function Arg(args, name) {
	if (!(name in args) || args[name] == null) return Failure("missing the argument " + name);
	return args[name];
}

function Opt(args, name, default_value) {
	return (name in args && args[name] != null) ? args[name] : default_value;
}

/// A cargo, by its label (`PASS`, `COAL`, `OIL_`, …, in any case, the underscore optional)
/// or by its name.
function Cargo(what) {
	local want = what.toupper();
	foreach (c, _ in GSCargoList()) {
		if (GSCargo.GetCargoLabel(c) == want || CargoLabel(c) == want || GSCargo.GetName(c).toupper() == want) return c;
	}
	return Failure("no cargo " + what + "; `cargos` lists them");
}

/// `args[name]` as a cargo, or null if it is not given.
function OptCargo(args, name) {
	return (name in args && args[name] != null) ? Cargo(args[name]) : null;
}

function CargoLabel(c) {
	local label = GSCargo.GetCargoLabel(c);
	while (label.len() > 0 && label.slice(-1) == "_") label = label.slice(0, -1);
	return label;
}

function Passengers() {
	foreach (c, _ in GSCargoList()) {
		if (GSCargo.HasCargoClass(c, GSCargo.CC_PASSENGERS)) return c;
	}
	return null;
}

function DateString(date) {
	local m = GSDate.GetMonth(date), d = GSDate.GetDayOfMonth(date);
	return GSDate.GetYear(date) + "-" + (m < 10 ? "0" : "") + m + "-" + (d < 10 ? "0" : "") + d;
}

/// The four neighbours' offsets.
function Offsets() {
	local sx = GSMap.GetMapSizeX();
	return [1, -1, sx, -sx];
}

function Inside(tile) {
	if (!GSMap.IsValidTile(tile)) return false;
	local x = GSMap.GetTileX(tile), y = GSMap.GetTileY(tile);
	return x > 0 && y > 0 && x < GSMap.GetMapSizeX() - 1 && y < GSMap.GetMapSizeY() - 1;
}

/// Every tile within `radius` of `center`, as a square, nearest first.
function TilesAround(center, radius) {
	local list = GSTileList();
	local cx = GSMap.GetTileX(center), cy = GSMap.GetTileY(center);
	local x0 = max(1, cx - radius), y0 = max(1, cy - radius);
	local x1 = min(GSMap.GetMapSizeX() - 2, cx + radius), y1 = min(GSMap.GetMapSizeY() - 2, cy + radius);
	list.AddRectangle(GSMap.GetTileIndex(x0, y0), GSMap.GetTileIndex(x1, y1));
	list.Valuate(GSMap.DistanceManhattan, center);
	list.Sort(GSList.SORT_BY_VALUE, true);
	return list;
}

/// A tile, a station, a town or an industry, as the tile it stands for.
function Place(where) {
	if (typeof where != "table") return Failure("say where: a tile, a station, a town or an industry");
	if ("tile" in where && where.tile != null) return Tile(where.tile);
	if ("station" in where && where.station != null) {
		if (!GSStation.IsValidStation(where.station)) return Failure("no station " + where.station);
		return GSStation.GetLocation(where.station);
	}
	if ("town" in where && where.town != null) {
		if (!GSTown.IsValidTown(where.town)) return Failure("no town " + where.town);
		return GSTown.GetLocation(where.town);
	}
	if ("industry" in where && where.industry != null) {
		if (!GSIndustry.IsValidIndustry(where.industry)) return Failure("no industry " + where.industry);
		return GSIndustry.GetLocation(where.industry);
	}
	return Failure("say where: a tile, a station, a town or an industry");
}

/// `args[name]` as a place, or null if it is not given.
function OptPlace(args, name) {
	return (name in args && args[name] != null) ? Place(args[name]) : null;
}

function StationSummary(s) {
	return {id = s, name = GSStation.GetName(s), tile = XY(GSStation.GetLocation(s))};
}

function Contains(array, value) {
	foreach (x in array) if (x == value) return true;
	return false;
}

/// Highest first, for `sort`.
function ByRank(a, b) {
	if (a.rank > b.rank) return -1;
	return a.rank < b.rank ? 1 : 0;
}
