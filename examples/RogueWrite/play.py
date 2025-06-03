# ruff: noqa: E501
from ailoy import Agent, Runtime
from rich.console import Console
from rich.panel import Panel

from levels import Character, Level
from settings import (
    ANNOUNCEMENT_BOSS,
    ANNOUNCEMENT_CLEAR,
    ANNOUNCEMENT_CLEAR_LEVEL,
    ANNOUNCEMENT_FAIL,
    ANNOUNCEMENT_INPUT,
    ANNOUNCEMENT_LONG_ABILITY,
    ANNOUNCEMENT_START,
    CONTENT_STAT,
    LEVEL_CONTENTS,
    MAX_LEN_ABILITY,
    PROMPT_CHECK_BATTLE,
    PROMPT_CHECK_INPUT,
    PROMPT_CLEAR_LEVEL,
    PROMPT_INPUT_ABILITY,
    PROMPT_INPUT_NAME,
    SYSTEM_MESSAGE,
    TITLE_BATTLE_RESULT,
    TITLE_CHECK_INPUT,
    TITLE_STAT,
)


def get_character_from_user(_console):
    print = _console.print
    while True:
        print(ANNOUNCEMENT_INPUT)
        name = input(PROMPT_INPUT_NAME).strip()
        ability = input(PROMPT_INPUT_ABILITY).strip()
        panel = Panel(
            ability,
            title=TITLE_CHECK_INPUT.format(name=name),
            title_align="center",
        )
        print()
        print(panel)

        if len(ability) > MAX_LEN_ABILITY:
            print(
                ANNOUNCEMENT_LONG_ABILITY.format(
                    MAX_LEN_ABILITY=MAX_LEN_ABILITY, len_ability=len(ability), ability=ability
                )
            )
            continue

        check = input(PROMPT_CHECK_INPUT.format(name=name, ability=ability)).strip()
        if check in ("", "y", "Y", "yes", "Yes"):
            break

    return Character(name, ability)


def main():
    _console = Console(highlight=False)
    print = _console.print

    print("\n[bold][italic]< [red]RogueWrite[/red] >[/italic][bold]\n")
    print(ANNOUNCEMENT_START)

    levels = [Level(i, **content) for i, content in enumerate(LEVEL_CONTENTS)]

    rt = Runtime()
    agent = Agent(rt, model_name="Qwen/Qwen3-8B", system_message=SYSTEM_MESSAGE)

    character = get_character_from_user(_console)
    for i, level in enumerate(levels):
        level.set_agent(agent)
        print(ANNOUNCEMENT_BOSS.format(level_index=i + 1))
        panel = Panel(
            level.boss.ability,
            title=f'[bold][italic]{level.boss.name}[/italic][/bold] — [blue][bold][italic]"{level.enter_message}"[/italic][/bold][/blue] —',
            title_align="center",
        )
        print(panel)

        input(PROMPT_CHECK_BATTLE)

        result = level.challenge(challenger=character)
        print()

        panel = Panel(
            result.get("unfolding"),
            title=TITLE_BATTLE_RESULT.format(level=level),
            title_align="left",
        )
        print(panel)
        print()

        if result.get("winner") == character.name:
            print(
                f'[bold][italic]{level.boss.name}[/italic][/bold] — [blue][bold][italic]"{level.clear_message}"[/italic][/bold][/blue] —'
            )
            print(ANNOUNCEMENT_CLEAR_LEVEL.format(character=character, level=level))
            input(PROMPT_CLEAR_LEVEL)
            continue
        else:
            print(
                f'[bold][italic]{level.boss.name}[/italic][/bold] — [blue][bold][italic]"{level.fail_message}"[/italic][/bold][/blue] —'
            )
            print(ANNOUNCEMENT_FAIL.format(character=character))
            break
    else:
        print(ANNOUNCEMENT_CLEAR.format(character=character))
        i += 1

    panel = Panel(
        CONTENT_STAT.format(character=character, level_index=i, max_level=len(levels)),
        title=TITLE_STAT.format(character=character),
        title_align="left",
    )
    print(panel)
    print("[bold][italic]> The End. <[/italic][bold]\n")

    agent.delete()
    rt.stop()


if __name__ == "__main__":
    main()
