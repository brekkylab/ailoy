## Picks

(none — see below)

## Why there is no shortlist

The must-haves for this posting are narrow and specific:

- **Solidity in production**: contracts the candidate wrote that held customer funds
- **Familiarity with common vulnerability classes** and how they are found (i.e. security-review practice)
- **3+ years of engineering experience**

Nice-to-haves (formal verification/fuzzing, EVM internals, prior audit engagement) were
deliberately kept out of the gating step per instructions — they were only meant to rank
people who already cleared the must-haves.

### What I searched

1. `headhunting distribution skill` (full list, 49 spellings) — no skill resembling
   Solidity, EVM, Vyper, smart contracts, or any blockchain-dev tooling exists on any
   profile in the pool. The list is dominated by web/mobile/data/ML/infra stacks
   (Python, Kubernetes, TypeScript, React, Rust, etc.).
2. `headhunting distribution title` and `distribution company` for "solidity",
   "blockchain", "smart", "contract", "chain" — zero matches on both axes.
3. `headhunting search --mentions <term>` (which scans headline, summary, titles,
   descriptions, and skills together, not just the skill list) for a wide sweep of
   domain vocabulary: `solidity`, `blockchain`, `smart contract`, `evm`, `defi`, `token`,
   `wallet`, `gas`, `fuzz`, `formal verification`, `audit`, `vyper`, `hardhat`, `truffle`,
   `openzeppelin`, `ethers`, `web3`, `nft`, `dao`, `proxy`, `upgradeable`,
   `storage layout`, `migration`, `ledger`, `settlement`, `custody`, `on-chain`,
   `protocol`, `mainnet`, `testnet`, `erc20`/`erc-20`/`erc721`, `cryptography`,
   `validator`, `consensus`.
4. The only hits with any blockchain flavor were adjacent roles, and reading their full
   position descriptions (`headhunting read`) rules every one of them out explicitly, in
   their own words:
   - **urn:li:person:l4wq9jkb** (Piper Voss, Frontend Engineer · dApp interfaces,
     Draftwell Dynamics): "I do not write contracts — I build the layer that makes
     interacting with them survivable... Contract development sits with a separate
     team; my work stops at the RPC boundary." Frontend/TypeScript/WebAssembly, not
     Solidity.
   - **urn:li:person:x6vd3ntq** (Jade Grantham, SRE · validator node operations,
     Junipex Dynamics): runs validator infrastructure (Kubernetes, Linux, Terraform,
     key handling) — infra/ops, not contract authorship. "My side of the stack is
     operations."
   - **urn:li:person:m3tb8vqd** (Tatum Hollis, Data Engineer · on-chain analytics,
     Quantile Dynamics): builds ingestion/decoding pipelines over block data in
     SQL/Python/Spark. "I work downstream of contracts, not on them." No contact on
     file for this person either.
   - **urn:li:person:r7nk4qwj** (Alex Merrick, Backend Engineer · fintech payments, "web3
     curious", Finlogic Labs): Java/PostgreSQL/Kafka payments backend engineer who
     "follows what happens in web3 with interest" — headline bait, no contract work of
     any kind, let alone Solidity in production.
5. No `certification` values relate to blockchain/security audit certs either (the
   certification list is all cloud/PM/ML/K8s certs).
6. Checked `job_function` distribution and `industry` values for the whole pool — only
   Backend / Frontend / ML / Data / Mobile / Infrastructure, and Information Technology /
   Internet / Computer Software / Financial Services. No blockchain- or smart-contract
   specific function or industry exists.

### Conclusion

The pool (600 candidates) does not contain anyone who has written or audited Solidity
contracts, let alone one who has shipped contracts that held funds. The closest profiles
are explicit about sitting adjacent to that work (frontend on top of contracts, infra
below them, or downstream analytics) and none can be read as meeting the core
must-have. Per instructions, a candidate who clearly fails a must-have does not make the
top 5, and an empty shortlist is preferable to including a near-miss that doesn't meet the
bar.

<!-- rejected -->

- **urn:li:person:l4wq9jkb** — Piper Voss — Frontend engineer for a "protocol" product,
  explicitly states she does not write contracts; wrong domain and wrong must-have.
- **urn:li:person:x6vd3ntq** — Jade Grantham — Validator-node SRE; strong infra/ops
  background but no contract-writing experience at all.
- **urn:li:person:m3tb8vqd** — Tatum Hollis — On-chain analytics data engineer,
  explicitly downstream of contracts; also has no contact method on file.
- **urn:li:person:r7nk4qwj** — Alex Merrick — Fintech backend engineer (Java, settlement/
  ledger domain experience is real) but "web3 curious" is headline framing only; no
  Solidity or contract work anywhere in the record.
