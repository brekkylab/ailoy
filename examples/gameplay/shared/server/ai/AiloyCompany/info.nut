// The agent's company. It does nothing itself: an AI is the one way to open a company on a
// dedicated server, and the bridge Game Script builds for it, in `GSCompanyMode`.
class AiloyCompany extends AIInfo {
	function GetAuthor()      { return "ailoy"; }
	function GetName()        { return "AiloyCompany"; }
	function GetShortName()   { return "ALOY"; }
	function GetDescription() { return "The company an ailoy agent runs through the AiloyBridge Game Script. Does nothing by itself."; }
	function GetVersion()     { return 1; }
	function GetDate()        { return "2026-09-29"; }
	function CreateInstance() { return "AiloyCompany"; }
	function GetAPIVersion()  { return "14"; }
	function UseAsRandomAI()  { return false; }
}

RegisterAI(AiloyCompany());
