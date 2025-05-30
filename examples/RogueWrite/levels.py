import json

from ailoy import Agent

from settings import ANNOUNCEMENT_IN_BATTLE, MAX_LEN_ABILITY


class Character:
    def __init__(self, name, ability):
        if len(ability) > MAX_LEN_ABILITY:
            raise ValueError(f"Ability is too long: {len(ability)} > {MAX_LEN_ABILITY} ({ability})")
        self.name = name
        self.ability = ability

    def __str__(self):
        return f"name:{self.name}\nability:{self.ability}\n"


class Boss(Character):
    def __init__(self, name, ability, level_index):
        super().__init__(name, ability)
        self.level_index = level_index


class Level:
    def __init__(
        self,
        index: int,
        boss_name: str,
        boss_ability: str,
        enter_message: str,
        clear_message: str,
        fail_message: str,
    ):
        self.index = index
        self.boss = Boss(boss_name, boss_ability, index)
        self.enter_message = enter_message
        self.clear_message = clear_message
        self.fail_message = fail_message

    def set_agent(self, agent: Agent):
        self.agent = agent

    def challenge(self, challenger: Character):
        prompt = f"{str(self.boss)}\n\n{str(challenger)}"

        for _ in range(10):
            output = ""
            for resp in self.agent.query(prompt):
                if resp.type == "output_text":
                    output += resp.content

            try:
                result = json.loads(output)
            except json.JSONDecodeError:
                print(ANNOUNCEMENT_IN_BATTLE)
                continue
            else:
                break
        else:
            raise RuntimeError("Battle ends with improper result.", output)

        return result
