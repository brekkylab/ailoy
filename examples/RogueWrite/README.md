# RogueWrite

Text-based 'rogue-lite' game inspired by '[Text Battle](https://plan9.kr/battle/)'.

You create your own character to defeat all the bosses.

Just try to play!

## How to play

1. (optional) Set your language. (refer to [Language settings](#language-settings))
2. Run `play.py` (refer to [Run](#run) section.)
3. Describe your own character by text.
4. By the level, a boss will appear, you can challenge it and check the results. (Just press enter to proceed!)

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
