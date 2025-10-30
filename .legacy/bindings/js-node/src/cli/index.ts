#!/usr/bin/env node
import { Command } from "commander";

import modelCommand from "./model";

const program = new Command();
program.name("ailoy").description("Ailoy CLI");
program.addCommand(modelCommand);

program.parse();

export default program;
