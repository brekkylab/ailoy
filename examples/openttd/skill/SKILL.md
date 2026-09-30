---
name: openttd
description: Run a transport company in a live game of OpenTTD — find towns and industries, build stations, roads and airports, buy vehicles and give them orders, let time pass and see what earns. Use it to play the game the console is running.
---

# OpenTTD

A game of [OpenTTD](https://www.openttd.org) is running, and one company in it is yours: **Ailoy Transport**. You play it with `ttd.py`, from this directory:

```sh
python3 ttd.py status
python3 ttd.py COMMAND -h      # what a command takes
```

The game stands **paused** between your commands. It runs only while a command is carried out
(a few game hours) and while `wait` lets time pass, so take as long as you like to think.
Someone may be watching the game: the view follows what you build.

## Places

A tile is `X,Y`. The map is a grid of tiles, `status` gives its size. Where a command takes a
place, it is a tile or one of `town:ID`, `industry:ID`, `station:ID`, the ids that `towns`,
`industries` and `stations` list. Distances are in tiles, counted along X plus along Y.

## Commands

**Looking**

| Command | What it gives |
|---|---|
| `status` | the date, money, loan, company value, this and last quarter's income and expenses |
| `report` | `status`, and the problems: idle or losing vehicles, cargo piling up, poor ratings, stations without vehicles |
| `towns [--near PLACE] [--limit N]` | towns, largest first or nearest first: population, passengers a month, your rating there |
| `industries [--cargo C] [--accepting] [--near PLACE]` | industries producing `C` (or accepting it): production last month and how much of it was carried |
| `cargos` | the cargos, and what each pays |
| `map PLACE [--radius R]` | the tiles around a place, a character each, with the towns, industries and your stations in view |
| `tile X,Y` | what is on a tile, and what a stop there would accept and supply |
| `look PLACE` | turn the view to a place and take a screenshot, saved in `/artifacts/shots/`: read it to see the place |
| `engines [--type road\|air] [--cargo C]` | the vehicles on sale now: capacity, speed, price, running cost |
| `airports` | the airport types, and which can be built this year |
| `stations` | your stations: what waits at each, its rating for each cargo, how many vehicles call |
| `vehicles [--station ID] [--ids ID,…]` | your vehicles: state, load, profit this year and last, orders |

**Building** — each finds the place itself when you give `--near`, and says what it cost.

| Command | What it does |
|---|---|
| `station --cargo C --near PLACE [--mode pickup\|dropoff\|both] [--kind bus\|truck]` | a road stop where it best picks up or drops off `C`: on a straight road it stops on the road, else beside one, else on bare land facing where a road can come in. Near `industry:ID` it only picks places in reach of that industry. `--tile X,Y` puts it exactly there |
| `road FROM TO` | finds a road between two places and builds it, over existing roads where it can, with bridges and tunnels where it must. A station as an end is joined at its entrance |
| `depot PLACE` | a road depot beside a road near a place: where road vehicles are bought and serviced |
| `airport --near PLACE [--type small]` | an airport where it reaches the most passengers, levelling land if it must. Comes with a hangar |
| `demolish X,Y` | clears a tile |

**Vehicles and money**

| Command | What it does |
|---|---|
| `buy --depot X,Y --engine ID [--count N] [--cargo C] --orders STOP …` | buys vehicles at a depot or hangar, refits them to `C`, gives the first the orders and the rest the same orders (shared), and starts them |
| `orders ID[,ID…] STOP …` | replaces vehicles' orders (`--share-with ID` to share another's) |
| `vehicle start\|stop\|depot\|sell\|clone ID[,ID…] [--count N]` | `sell` sends it to a depot and sells it there; `clone` buys copies sharing its orders |
| `loan AMOUNT\|max\|min` | sets the loan, in steps of `loan_step` |
| `wait DAYS` | lets the game run for 1 to 366 days, then gives the `report` |
| `save NAME` | saves the game |

A **stop** in orders is `STATION` or `STATION:flag+flag`. The flags: `full` (wait for a full
load of any cargo), `full_all`, `unload`, `transfer`, `noload`, `nounload`. `12:full 7` loads
fully at station 12 and delivers at 7, which is the usual order for freight.

Money is in pounds. Vehicles you buy are renewed on their own when they grow old.

## What earns

You start with a loan and some money. Nothing earns until vehicles run, so spend early: take
the loan to `max`, build a first route, and pay the loan back from what it earns.

* **Coal to a power station by truck** is the classic start. Pick a mine producing 100 t a month
  or more and a power station 20–50 tiles away, not much further: a truck carries 20 t. Build
  the pickup stop near the mine and the drop-off stop near the power station, a road between
  them, a depot beside the road near the mine, and three or four trucks with orders
  `PICKUP:full DROP`. Any freight works so, where the industry that takes it is near: iron ore
  to a steel mill, wood to a sawmill, livestock and grain to a factory, oil to a refinery.
* **Passengers by air** pay best of all over long distances. Two large towns (1000 people or
  more) 80 tiles or more apart, a small airport near each, and two planes with orders `A B`.
  A plane costs about as much as six trucks: count your money first. In a small airport,
  only small planes: big ones crash there.
* **Buses between towns** are cheap and earn little, but make a town like you: a town that
  does not like you will not let you build there. A stop in each town (`--mode both`), a
  road, a depot, two buses with orders `A B`. Buses do not want `full`: both ends supply
  passengers.

Then watch it. After a month or two, `report` and `stations` say what to do: cargo piling up
at a pickup wants more vehicles (`vehicle clone`); vehicles queueing to load want fewer; a
vehicle that earns nothing is on a route that does not work. The rating a station has for a
cargo is the share of what is produced it gets: it falls when cargo waits long, so keep
vehicles calling. Make sure a new route earns before you build the next.

A route's income is roughly what it carries times the distance: a longer route earns more a
trip, but takes longer. Freight only earns where it is accepted: `tile` and `station` say what
a stop accepts, and a drop-off stop must accept the cargo.

## The map

`map` draws one character a tile, X growing left to right and Y top to bottom:

```
. flat land   , sloped land   # road   = rail   ~ water   h town building   I industry
S your station   s another's   D your depot   d another's   B bridge   T tunnel   x other
```

Land (`.` and `,`) can be built on. Houses, industries and water cannot, and roads go round,
over or under them. On screen the same map is drawn in perspective, turned 45 degrees: X runs
down to the left and Y down to the right.

## When something fails

Every command says what went wrong, in OpenTTD's words when it has them (`ERR_LOCAL_AUTHORITY_REFUSES`,
`ERR_NOT_ENOUGH_CASH`, `ERR_AREA_NOT_CLEAR`, …).

* A stop or an airport with no place near: search further (`--search 12`), or pick another
  end.
* A road that is not found: the ends are cut off by water or the map's edge, or it is too far
  to search. Try other ends, or a road in two halves through a place between.
* A town that refuses: your rating there is too low, often from trees you cut. Build
  elsewhere for a while, or run buses there.
* Not enough money: `status` says what you have and what you may still borrow.

## Keeping track

Keep notes in `/artifacts/notes.md` as you go: each route with its stations, depot and
vehicles, what it earns, and what you mean to do next. Older command output drops out of what
you remember as the game goes on, and the notes are how you pick up where you left off.
