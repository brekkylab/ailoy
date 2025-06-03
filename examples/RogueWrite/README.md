# RogueWrite

Text-based _'rogue-lite'_ game inspired by '[Text Battle](https://plan9.kr/battle/)'.  
Create your own character to defeat all the bosses!

[play-en.webm](https://github.com/user-attachments/assets/de60ecca-9d95-4242-a797-91aadf24b5e8)

<details>
<summary>한국어 플레이 영상 보기</summary>

[play-ko.webm](https://github.com/user-attachments/assets/234b15bc-fc94-44fe-8876-8b3081bbd0de)

</details>

**_It's your turn. Just try to play!_**

## How to Play

1. _(optional)_ Set your language. (refer to [Language settings](#language-settings))
2. Run `play.py` (refer to [Run](#run) section.)
3. Describe your own character by text.
4. Challenge the boss that appears on each level and check the results. (Just press enter to proceed!)

## How to Run

### with `uv`

```bash
$ uv init
$ uv run play.py
```

### without `uv`

```bash
$ pip install .
$ python play.py
```

## Language settings
Currently, RogueWrite supports **_English_** and **_Korean_**.

`ROGUEWRITE_LANGUAGE`: Environment variable to set the language. Options: [`en`, `ko`], default is `en`.

### English

The default language is English without any setting.

or just set the environment variable explicitly:

```
ROGUEWRITE_LANGUAGE=en
```

### Korean

```
ROGUEWRITE_LANGUAGE=ko
```
