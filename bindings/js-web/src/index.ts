import AiloyModule from "./ailoy_js_web";

async function getModule() {
  return await AiloyModule();
}

window.getModule = getModule;

export { getModule };
