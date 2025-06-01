# RogueWrite

Text-based 'rogue-lite' game inspired by '[Text Battle](https://plan9.kr/battle/)'
Create your own character to defeat all the bosses.


https://github.com/user-attachments/assets/872ad8e6-980c-4cff-b7f9-ada556736557


<details>
<summary>한국어 플레이 영상</summary>


https://github.com/user-attachments/assets/9e2c5c74-3a02-4c0c-b895-0afccd68c43c


</details>

**_Just try to play!_**

## How to play

1. (optional) Set your language. (refer to [Language settings](#language-settings))
2. Run `play.py` (refer to [Run](#run) section.)
3. Describe your own character by text.
4. Challenge the boss that appears on each level and check the results. (Just press enter to proceed!)

## Run

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
