-- The domain knowledge lives here (spec §1.1). These 26 lines that strip overlapping
-- employment are a recruiting judgment, not a database feature, so they sit in the
-- example's VIEW rather than in `cortex-execs/sqlite`. The agent writes only
-- `SELECT * FROM candidate_tenure WHERE real_years >= 4`.
--
-- The difficulty does not disappear, it moves. Sum the spans without the view and the
-- answer is still wrong — that is trap 3.
--
-- The CTE is in four stages because SQLite refuses a window function nested inside an
-- aggregate (`misuse of window function MAX()`). Sparing the agent non-obvious
-- constraints like that every turn is part of why the view exists.
--
-- `2026*12+8` has to match `AS_OF` in `sql/load.py`, which checks it on every load.
-- `assert_as_of_matches` in `load.py` checks it — a mismatch is silently wrong.
CREATE VIEW candidate_tenure AS
WITH span AS (
  SELECT candidate_id AS id,
         start_year*12 + COALESCE(start_month,1) AS s,
         COALESCE(end_year*12 + COALESCE(end_month,1), 2026*12+8) AS e
  FROM positions
),
scan AS (
  SELECT id, s, e,
         MAX(e) OVER (PARTITION BY id ORDER BY s
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
  FROM span
),
island AS (
  SELECT id, s, e,
         SUM(CASE WHEN prev_end IS NULL OR s > prev_end THEN 1 ELSE 0 END)
           OVER (PARTITION BY id ORDER BY s) AS grp
  FROM scan
),
merged AS (SELECT id, grp, MIN(s) AS s, MAX(e) AS e FROM island GROUP BY id, grp)
SELECT c.id, c.first_name || ' ' || c.last_name AS name,
       (SELECT SUM(e-s) FROM span WHERE span.id = c.id) AS naive_months,
       SUM(m.e - m.s) AS real_months,
       ROUND(SUM(m.e - m.s)/12.0, 1) AS real_years
FROM merged m JOIN candidates c ON c.id = m.id
GROUP BY c.id, name;

-- Confines the convention that `end_year IS NULL` means current to one place. No separate
-- boolean, following the README's original design (spec §2.0); with one value there is
-- less to keep in sync.
--
-- **It has to be one row per candidate.** 17 people measurably hold more than one current
-- role — that is concurrent employment, and it is the material for trap 3. Because
-- `candidate_brief` `LEFT JOIN`s this view, emitting several rows would list those 17
-- twice in the scan. That actually happened: a 600-person database whose
-- `candidate_brief` had 617 rows.
--
-- It fails quietly. The list looks right, the headcount inflates, and `--limit 50` pushes
-- someone off without a word. The one pushed off is a concurrently employed person — that
-- is, one of the trap set.
--
-- The most recently started role is taken as representative. The concurrency itself
-- remains as the gap between `naive_months` and `real_months` in `candidate_tenure`, and
-- the question this view answers is only "where do they work now".
CREATE VIEW current_position AS
SELECT p.candidate_id AS id, p.title, p.company_name, p.company_size,
       p.employment_type, p.workplace_type, p.location,
       p.start_year, p.start_month
FROM positions p
WHERE p.end_year IS NULL
  AND p.ord = (SELECT q.ord FROM positions q
               WHERE q.candidate_id = p.candidate_id AND q.end_year IS NULL
               ORDER BY q.start_year DESC, q.start_month DESC, q.ord
               LIMIT 1);

-- The first two stand in for arithmetic; these three show the terrain of the data
-- (spec §2.2).
--
-- Why they are needed: spelling drifts (spec §3.4). `Rust` appears as `rust-lang`,
-- `Rust Lang`, `Async Rust`, `Tokio`, and `러스트`, and FTS5's `MATCH 'rust'` recalls
-- 85–95% of them — so 5–15% are not found. An agent looking at the terrain through these
-- views before widening its queries is what DATASET_PLAN §2.2 asks for.
CREATE VIEW skill_distribution AS
SELECT name, COUNT(*) AS holders
FROM skills GROUP BY name ORDER BY holders DESC;

CREATE VIEW title_distribution AS
SELECT title, COUNT(*) AS holders
FROM positions GROUP BY title ORDER BY holders DESC;

CREATE VIEW location_distribution AS
SELECT location, COUNT(*) AS positions_here
FROM positions GROUP BY location ORDER BY positions_here DESC;

-- **The narrow view. The agent's first scan uses this one.**
--
-- The `sqlite` command caps rows (`--limit`) and not bytes (Plan A). So
-- `SELECT * FROM candidates LIMIT 100` pours 100 long `summary` fields into the context —
-- tens to hundreds of KB. The layer split (spec §1.1) already answers this: the domain
-- layer supplies a narrow view and the instruction forbids `SELECT *`.
--
-- Adding a byte cap to the command is not an obvious improvement — it would create a
-- second truncation axis and a second notation to report it.
--
-- The point of this view is that `summary` and `description` are absent. Reading those is
-- the second pass (spec §5.3, step 4), and that is done one person at a time.
--
-- `contact_rows` of 0 means there is no way to reach them — trap 12. The `LEFT JOIN` is
-- there because some people have no current role (every position ended), and then
-- `current_title` is NULL.
CREATE VIEW candidate_brief AS
SELECT c.id, c.first_name || ' ' || c.last_name AS name,
       c.headline, c.city, c.country, c.seniority, c.job_function,
       c.profile_language, c.open_to_work, c.last_updated_at,
       t.real_years,
       cp.title AS current_title, cp.company_name AS current_company,
       cp.company_size AS current_company_size,
       (SELECT COUNT(*) FROM contacts x WHERE x.candidate_id = c.id) AS contact_rows
FROM candidates c
LEFT JOIN candidate_tenure t ON t.id = c.id
LEFT JOIN current_position cp ON cp.id = c.id;
