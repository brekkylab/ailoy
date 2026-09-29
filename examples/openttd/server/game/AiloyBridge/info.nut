class AiloyBridge extends GSInfo {
	function GetAuthor()      { return "ailoy"; }
	function GetName()        { return "AiloyBridge"; }
	function GetShortName()   { return "ALYB"; }
	function GetDescription() { return "Takes commands as JSON from the admin port and carries them out for the agent's company."; }
	function GetVersion()     { return 1; }
	function GetDate()        { return "2026-09-29"; }
	function CreateInstance() { return "AiloyBridge"; }
	function GetAPIVersion()  { return "14"; }
	function IsDeveloperOnly(){ return false; }
}

RegisterGS(AiloyBridge());
