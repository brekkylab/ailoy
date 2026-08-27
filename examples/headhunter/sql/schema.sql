-- Mirrors what the LinkedIn Recruiter product UI shows a recruiter (spec §2.0).
-- Not any LinkedIn API: there is no prospecting search API, and `openToWork` is a 2020 UI
-- feature that has never been exposed as a field on any official API.
-- The comment after each column names the corresponding Recruiter search filter.

CREATE TABLE candidates (
  id TEXT PRIMARY KEY,              -- urn:li:person:aBcD1234
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,          -- First names / Last names
  headline TEXT NOT NULL,           -- where trap 1 (headline bait) lives
  summary TEXT NOT NULL,            -- filled by the narration layer
  city TEXT NOT NULL,
  country TEXT NOT NULL,            -- Locations
  industry TEXT NOT NULL,           -- Industries
  job_function TEXT NOT NULL,       -- Job functions
  seniority TEXT NOT NULL,          -- Seniority. The basis for judging trap 13
  profile_language TEXT NOT NULL,   -- Profile languages. Trap 10
  open_to_work INTEGER NOT NULL,    -- Open to work (Spotlight). Trap 7
  connections_count INTEGER NOT NULL,
  last_updated_at TEXT NOT NULL,    -- trap 8 (stale profile)
  public_profile_url TEXT NOT NULL
);

CREATE TABLE positions (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  ord INTEGER NOT NULL,             -- JSON array order, aligned with the answer key's descriptions
  title TEXT NOT NULL,              -- Job titles. current vs past is decided by end_year
  company_name TEXT NOT NULL,       -- the spelling drifts (spec §3.4)
  company_urn TEXT NOT NULL,        -- the urn stays the same through the drift
  company_size TEXT NOT NULL,       -- Company sizes. Trap 13's other axis
  employment_type TEXT NOT NULL,    -- the basis for trap 3 (concurrent employment)
  workplace_type TEXT NOT NULL,     -- trap 6
  location TEXT NOT NULL,
  description TEXT NOT NULL,        -- filled by the narration layer
  start_year INTEGER NOT NULL,
  start_month INTEGER NOT NULL,
  end_year INTEGER,
  end_month INTEGER,                -- NULL = current
  PRIMARY KEY (candidate_id, ord)
);

CREATE TABLE skills (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  name TEXT NOT NULL,
  endorsement_count INTEGER NOT NULL
);

CREATE TABLE educations (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  school_name TEXT NOT NULL,
  degree_name TEXT NOT NULL,
  field_of_study TEXT NOT NULL,
  start_year INTEGER NOT NULL,
  end_year INTEGER NOT NULL
);

CREATE TABLE certifications (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  name TEXT NOT NULL,
  authority TEXT NOT NULL
);

CREATE TABLE languages (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  name TEXT NOT NULL,
  proficiency TEXT NOT NULL         -- NATIVE_OR_BILINGUAL and the like
);

-- `openToWork` is not one boolean. It has sub-fields the candidate fills in that only
-- recruiters see (Recruiter documentation). That structure matters when handling trap 7
-- (a strong fit with open_to_work=false), so it is its own table.
CREATE TABLE open_to_work_prefs (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  desired_title TEXT NOT NULL,      -- up to five
  location_type TEXT NOT NULL,      -- On-site / Remote / Hybrid
  desired_location TEXT NOT NULL,
  start_date TEXT NOT NULL,
  employment_type TEXT NOT NULL
);

-- **Only people who can be reached have a row.** Trap 12 (no contact) is expressed as
-- the absence of one.
CREATE TABLE contacts (
  candidate_id TEXT NOT NULL REFERENCES candidates(id),
  method TEXT NOT NULL,             -- inmail / referral
  note TEXT NOT NULL
);

CREATE INDEX positions_by_candidate ON positions(candidate_id);
CREATE INDEX skills_by_candidate ON skills(candidate_id);
CREATE INDEX skills_by_name ON skills(name);

-- Search is FTS5. Why not embeddings (`sqlite-vec`) is in spec §2.3: measured coverage
-- and ranking matched FTS5 exactly, and it would push domain wiring inside a
-- general-purpose tool, breaking the layer split.
--
-- **It has to carry both `headline` and `skills`.** Plan B measured it: of the 65 core
-- candidates, 31 carry a `rust` token; 29 of them have it on both surfaces, but 2 have it
-- on only one —
--
--   headline only : the `headline-bait` trap (having the keyword only there is its definition)
--   skills only   : the `skills-without-evidence` trap (only in the list, by definition)
--
-- By definition neither has an alternative surface, so dropping one column takes that
-- trap out of the search results — and a search-stage trap that is not in the results is
-- not a trap. The other 29 survive either way, so **this failure is silent**: the data and
-- the index both look right and the result sizes are normal.
--
-- This still holds after the narration layer is filled. Other candidates gain a surface
-- through `description`, but these two cannot: `skills-without-evidence` is a trap
-- precisely because the keyword is **absent** from the descriptions, and `headline-bait`
-- because it is in the headline alone.
CREATE VIRTUAL TABLE candidate_fts USING fts5(
  id UNINDEXED,
  headline,
  summary,
  titles,        -- position titles joined by spaces
  descriptions,  -- position descriptions joined
  skill_names,   -- skill names joined
  tokenize = 'unicode61'
);
