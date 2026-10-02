// The bridge: commands come in as JSON from the admin port, and are carried out for the
// agent's company in `GSCompanyMode`.
//
// A request is `{id, cmd, args}`. The answer is `{id, ok, result}` or `{id, ok: false, error}`.
// A long list goes ahead of it in messages `{id, part: [...], done: false}`, since one
// message to the admin port holds no more than 9000 bytes of JSON.

require("util.nut");
require("road.nut");
require("commands.nut");

class AiloyBridge extends GSController {
	company = -1;
	// Vehicles sent to a depot to be sold there once they stop.
	to_sell = [];
	requests = [];

	function Start() {
		::BRIDGE <- this;
		GSLog.Info("AiloyBridge is up");
		local last_sell_check = 0;
		while (true) {
			this.Pump();
			if (this.requests.len() > 0) {
				local req = this.requests.remove(0);
				this.Serve(req);
			} else {
				this.Sleep(1);
			}
			if (GSController.GetTick() - last_sell_check > 20) {
				last_sell_check = GSController.GetTick();
				this.SellWaiting();
			}
		}
	}

	function Pump() {
		while (GSEventController.IsEventWaiting()) {
			local ev = GSEventController.GetNextEvent();
			if (ev == null) break;
			if (ev.GetEventType() == GSEvent.ET_ADMIN_PORT) {
				local req = GSEventAdminPort.Convert(ev).GetObject();
				if (req != null) this.requests.append(req);
			}
		}
	}

	function Serve(req) {
		local id = "id" in req ? req.id : null;
		local cmd = "cmd" in req ? req.cmd : "";
		local args = ("args" in req && req.args != null) ? req.args : {};
		if (!(cmd in Commands)) {
			GSAdmin.Send({id = id, ok = false, error = "no command " + cmd});
			return;
		}
		if (cmd != "setup" && GSCompany.ResolveCompanyID(this.company) == GSCompany.COMPANY_INVALID) {
			GSAdmin.Send({id = id, ok = false, error = "the company is gone"});
			return;
		}
		local mode = cmd == "setup" ? null : GSCompanyMode(this.company);
		GSRoad.SetCurrentRoadType(GSRoad.ROADTYPE_ROAD);
		local result = Commands[cmd](args);
		if (Failed(result)) {
			GSAdmin.Send({id = id, ok = false, error = result.message});
			return;
		}
		// Whoever watches is shown what was built, or what the agent looks at. Only the deity
		// may move everyone's view: dropping the company's mode goes back to it.
		mode = null;
		if (typeof result == "table" && "look_at" in result) {
			local tile = result.look_at;
			delete result.look_at;
			if (!GSViewport.ScrollEveryoneTo(tile)) GSLog.Warning("scrolling: " + GSError.GetLastErrorString());
		}
		// A list in `items` goes ahead in parts.
		if (typeof result == "table" && "items" in result) {
			local items = result.items;
			delete result.items;
			local i = 0;
			while (i < items.len()) {
				local part = items.slice(i, min(i + 20, items.len()));
				if (!GSAdmin.Send({id = id, part = part, done = false})) {
					// Too long even so: one at a time.
					foreach (item in part) GSAdmin.Send({id = id, part = [item], done = false});
				}
				i += 20;
			}
		}
		if (!GSAdmin.Send({id = id, ok = true, done = true, result = result})) {
			GSAdmin.Send({id = id, ok = false, error = "the answer was too long to send"});
		}
	}

	function SellWaiting() {
		if (this.to_sell.len() == 0) return;
		local mode = GSCompanyMode(this.company);
		local left = [];
		foreach (v in this.to_sell) {
			if (!GSVehicle.IsValidVehicle(v)) continue;
			if (GSVehicle.IsStoppedInDepot(v)) {
				if (!GSVehicle.SellVehicle(v)) left.append(v);
			} else {
				left.append(v);
			}
		}
		this.to_sell = left;
	}

	function Save() {
		return {company = this.company, to_sell = this.to_sell};
	}

	function Load(version, data) {
		if ("company" in data) this.company = data.company;
		if ("to_sell" in data) this.to_sell = data.to_sell;
	}
}
