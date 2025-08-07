import { _APIModel } from "./apiModel";
import { _LocalModel } from "./localModel";

type AiloyModel = _APIModel | _LocalModel;
export { type AiloyModel };
export * from "./apiModel";
export * from "./localModel";
