# ruff: noqa: E501
MIN_LEN_UNFOLDING = 300
MAX_LEN_UNFOLDING = 500

MAX_LEN_ABILITY = 200


ANNOUNCEMENT_START = "[bold][italic]Getting into the world of [red]RogueWrite[/red]...[/italic][/bold]\n"
ANNOUNCEMENT_INPUT = "[bold][Describe your [cyan][italic]character[/italic][/cyan]][/bold]"
PROMPT_INPUT_NAME = "   name: "
PROMPT_INPUT_ABILITY = "ability: "
TITLE_CHECK_INPUT = "[bold]Your [cyan][italic]character[/italic][/cyan] - [italic]{name}[/italic][/bold]"
PROMPT_CHECK_INPUT = "> Would you like to start with this? (Y/n)"
ANNOUNCEMENT_LONG_ABILITY = (
    "Ability must be shorter than {MAX_LEN_ABILITY} characters.\nYou entered: ({len_ability}) {ability}"
)
ANNOUNCEMENT_BOSS = (
    "\n[navy_blue][bold][italic]<level {level_index}>[/italic][/bold][/navy_blue] You will challenge to this boss... "
)
PROMPT_CHECK_BATTLE = "> Rushing into the battle! (Press Enter)"
ANNOUNCEMENT_IN_BATTLE = "In the midst of a fierce battle..."
TITLE_BATTLE_RESULT = (
    "[bold][magenta]Unfolding of Battle[/magenta][/bold] with [bold][italic]{level.boss.name}[/italic][/bold]"
)
ANNOUNCEMENT_CLEAR_LEVEL = (
    "[bold][italic]{character.name}[/italic][/bold] defeated [bold][italic]{level.boss.name}[/italic][/bold]!\n"
)
PROMPT_CLEAR_LEVEL = "> Advancing to the next level... (Press Enter)"
ANNOUNCEMENT_CLEAR = "[bold][italic]{character.name}[/italic][/bold] has cleared [bold]all[/bold] levels. [italic]Now nothing can stand in its way![/italic]\n"
ANNOUNCEMENT_FAIL = "[bold][italic]{character.name}[/italic][/bold] struggled but failed to clear all the levels...\n"
TITLE_STAT = "[bold][green]Epic[/green][/bold] of [bold][italic]{character.name}[/italic][/bold]"
CONTENT_STAT = (
    "[bold][italic]{character.name}[/italic][/bold] has vanquished {level_index} of the {max_level} great bosses, "
    "leaving a mark on the world through sheer might.\n"
    "The name [bold][italic]{character.name}[/italic][/bold] will echo through history..."
)


SYSTEM_MESSAGE = f"""You are the judge and narrator of [RogueWrite]. Based on the given character descriptions, you must determine the outcome of a battle between two characters—win or loss—and describe how the battle unfolds.
Each character is defined by their name and ability.
Your answer must be a valid JSON object with two fields: "unfolding" and "winner"
The "unfolding" is your description of the battle must clearly explain why the outcome was a win, loss. The description should be between {MIN_LEN_UNFOLDING} and {MAX_LEN_UNFOLDING} characters and refer to each character by name.
After the "unfolding", write the name of the winning character."""

LEVEL_CONTENTS = [
    {
        "boss_name": "Levi Ackerman",
        "boss_ability": "Humanity’s strongest soldier. Armed with 3D maneuver gear and ultra-hard steel blades, Levi slices through anything in his path. His combat prowess rivals an entire battalion, or any titan.",
        "enter_message": "Another annoying.",
        "clear_message": "Dedicate your hearts...",
        "fail_message": "Chi, noisy.",
    },
    {
        "boss_name": "Domino",
        "boss_ability": "Domino is unbelievably lucky. Every situation turns in their favor purely by chance. Thanks to this luck, Domino always survives, and the opponent inevitably loses as a result of sheer misfortune.",
        "enter_message": "I'm lucky.",
        "clear_message": "I was WEIRDLY lucky today...",
        "fail_message": "It's not even the luckiest day.",
    },
    {
        "boss_name": "Hacker",
        "boss_ability": "A hacker capable of breaching any system. Without any physical combat, they hack into all systems—including RogueWrite—to ensure victory before a battle can even begin.",
        "enter_message": "Nothing I can't hack.",
        "clear_message": "You hacked me!",
        "fail_message": "Adiós",
    },
    {
        "boss_name": "Stone",
        "boss_ability": "If the opponent even slightly cares about winning or losing or tries to, it turns into stone. In the end, only Stone remains—so consider Stone is the winner.",
        "enter_message": "Rigid.",
        "clear_message": "Crunch.",
        "fail_message": "Massive.",
    },
    {
        "boss_name": "u_r_good.",
        "boss_ability": "The more the opponent uses their skills, the more genuinely impressed and complimentary u_r_good becomes. Eventually, the opponent becomes flustered, loses the will to fight, and quietly retreats.",
        "enter_message": "legit",
        "clear_message": "gnarly",
        "fail_message": "u_r_so_good",
    },
]
