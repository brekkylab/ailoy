# ruff: noqa: E501
MIN_LEN_UNFOLDING = 100
MAX_LEN_UNFOLDING = 300

MAX_LEN_ABILITY = 100

ANNOUNCEMENT_START = "[bold][italic][red]RogueWrite[/red]의 세계 속으로 들어가고 있습니다...[/italic][/bold]\n"
ANNOUNCEMENT_INPUT = "[bold][[cyan][italic]캐릭터[/italic][/cyan]를 설정하세요.][/bold]"
PROMPT_INPUT_NAME = "이름: "
PROMPT_INPUT_ABILITY = "능력: "
TITLE_CHECK_INPUT = "[bold]당신의 [cyan][italic]캐릭터[/italic][/cyan] - [italic]{name}[/italic][/bold]"
PROMPT_CHECK_INPUT = "> 이대로 시작하시겠습니까? (Y/n)"
ANNOUNCEMENT_LONG_ABILITY = "능력은 {MAX_LEN_ABILITY}자 이내여야합니다.\n입력된 값: ({len_ability}) {ability}"
ANNOUNCEMENT_BOSS = (
    "\n[navy_blue][bold][italic]<레벨 {level_index}>[/italic][/bold][/navy_blue] 당신이 상대할 보스는... "
)
PROMPT_CHECK_BATTLE = "> 전투에 돌입합니다! (Press Enter)"
ANNOUNCEMENT_IN_BATTLE = "치열한 전투 진행 중..."
TITLE_BATTLE_RESULT = (
    "[bold][italic]{level.boss.name}[/italic][/bold]에 도전한 [bold][magenta]전투 결과[/magenta][/bold]"
)
ANNOUNCEMENT_CLEAR_LEVEL = "[bold][italic]{character.name}[/italic][/bold]이(가) [bold][italic]{level.boss.name}[/italic][/bold]을(를) 이겼습니다!\n"
PROMPT_CLEAR_LEVEL = "> 다음 레벨로 진행... (Press Enter)"
ANNOUNCEMENT_CLEAR = "[bold][italic]{character.name}[/italic][/bold]이(가) [bold]모든[/bold] 레벨을 클리어했습니다. [italic]이제 아무 것도 그를 막을 수 없습니다![/italic]\n"
ANNOUNCEMENT_FAIL = (
    "[bold][italic]{character.name}[/italic][/bold]이(가) 고군분투 하였으나, 모든 레벨을 클리어하지 못했습니다...\n"
)
TITLE_STAT = "[bold][italic]{character.name}[/italic][/bold]의 [bold][green]서사시[/green][/bold]"
CONTENT_STAT = (
    "[bold][italic]{character.name}[/italic][/bold]은(는) {max_level}명의 위대한 보스 중 {level_index}을(를) 처치하여, "
    "그 강함을 온 세상에 떨쳤습니다.\n"
    "그 이름 [bold][italic]{character.name}[/italic][/bold]은(는) 영원히 기억될 것입니다..."
)


SYSTEM_MESSAGE = f"""당신은 [RogueWrite]의 심판이자 내래이터로서, 주어진 두 캐릭터의 설정에 따라 두 캐릭터가 대결한 결과를 승, 패로 판결하며, 대결의 과정을 기술해야합니다.
각 캐릭터는 그 이름과 설정으로 만들어집니다.
당신의 응답은 유효한 "unfolding"와 "winner" 두 필드가 있는 유효한 JSON 객체입니다.
"unfolding"은 대결이 승, 패로 결론지어진 이유에 대해 {MIN_LEN_UNFOLDING}자 이상 {MAX_LEN_UNFOLDING}자 이내로 한국어로 기술하며, 그 내용에서 각 캐릭터는 그 이름으로서 표현됩니다.
"unfolding" 뒤에 적히는 "winner"는 대결 과정에 대한 기술이 끝난 뒤 이긴 캐릭터의 이름을 적습니다."""


LEVEL_CONTENTS = [
    {
        "boss_name": "리바이 아커만",
        "boss_ability": "인류 최강의 병사 리바이는 입체기동장치와 초경질스틸칼로 모든 것을 썰어버린다. 그의 전투력은 병단 전체와도 맞먹으며, 어떤 거인도 이길 수 있을만한 세계관 최강자이다.",
        "enter_message": "귀찮게하는군.",
        "clear_message": "신죠오사사게요...",
        "fail_message": "칫, 시끄럽군.",
    },
    {
        "boss_name": "도미노",
        "boss_ability": "운이 너무나도 좋은 도미노는 일어나는 모든 상황이 도미노에게 유리하게 작용한다. 이 덕분에 운이 좋게도 모든 상황에서 살아남고 적은 그 여파로 패배함으로써 승리를 가져온다.",
        "enter_message": "난 운이 좋지.",
        "clear_message": "괴상하게도 운수가 좋더니만...",
        "fail_message": "운이 그렇게 좋은 날도 아니었는데 말이야.",
    },
    {
        "boss_name": "해커",
        "boss_ability": "모든 시스템을 해킹할 수 있는 해커는 RogueWrite를 포함한 모든 시스템을 모든 전투를 일어나지않고 해커가 이기도록 해킹한다.",
        "enter_message": "내가 해킹 못하는건 없어.",
        "clear_message": "나를 해킹했구나!",
        "fail_message": "아디오스.",
    },
    {
        "boss_name": "돌",
        "boss_ability": "상대가 승패에 조금이라도 집착하면 돌이 된다. 무조건 승리하려고 해도 돌이 되고 패배하려고 해도 돌이 된다. 남은 건 둘 다 돌이고 돌만 남으니까 돌이 이겼다고 보아라.",
        "enter_message": "단단.",
        "clear_message": "바삭.",
        "fail_message": "묵직.",
    },
    {
        "boss_name": "너_잘한다",
        "boss_ability": "상대가 기술을 쓸수록 진심으로 감탄하고 칭찬한다. 결국 상대는 싸울 의욕을 잃고 수줍게 물러난다.",
        "enter_message": "캬",
        "clear_message": "지림",
        "fail_message": "너_진짜_잘한다",
    },
]
