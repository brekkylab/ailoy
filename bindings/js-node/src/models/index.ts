import APIModel, { APIModel as _APIModel } from "./apiModel";
import LocalModel, { LocalModel as _LocalModel } from "./localModel";

export type AiloyModel = _APIModel | _LocalModel;
export { APIModel, LocalModel };
