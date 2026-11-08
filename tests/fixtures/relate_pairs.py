"""Synthetic labelled pairs for the live Jev check. Labels were written by hand BEFORE any Jev call (no model chose them).
(older text, newer text, expected). Expected is what a careful reader says about NEWER relative to OLDER;
none = same topic but no relation of the four applies. Dates are fixed so older < newer."""
PAIRS = [
    # supersedes: newer replaces the older rule/decision
    ("The deploy script runs on Node 18 in the staging pipeline.", "The deploy script now runs on Node 22 in the staging pipeline; Node 18 is retired.", "supersedes"),
    ("Backups of the notes database go to /mnt/backup/daily every night at 02:00.", "Backups of the notes database moved to /mnt/archive/nightly at 03:30; the daily folder is no longer used.", "supersedes"),
    ("The API rate limit for free users is 60 requests per minute.", "The API rate limit for free users is now 120 requests per minute instead of 60.", "supersedes"),
    ("Pull requests need one approval before merge.", "Pull requests need two approvals before merge, replacing the one-approval rule.", "supersedes"),
    ("The team standup is at 09:30 on weekdays.", "The team standup moved to 10:00 on weekdays.", "supersedes"),
    ("Use the blue theme as the default colour for the dashboard.", "The dashboard default colour is now graphite; the blue theme is no longer the default.", "supersedes"),
    ("Logs are kept for 7 days in the ingestion service.", "The ingestion service keeps logs for 30 days, changed from 7.", "supersedes"),
    ("Release builds are signed with key A.", "Release builds are signed with key B since key A was rotated out.", "supersedes"),
    # contradicts: incompatible claims, neither retires the other by itself
    ("The cache layer improves read latency for the search service.", "The cache layer made read latency worse for the search service in our measurements.", "contradicts"),
    ("Parsing the config with regular expressions is safe for all inputs.", "Parsing the config with regular expressions is unsafe: it breaks on nested quotes.", "contradicts"),
    ("The nightly job always finishes within 10 minutes.", "The nightly job never finishes within 10 minutes; it takes about an hour.", "contradicts"),
    ("Dependency X has no known licence problems.", "Dependency X is under a licence that forbids commercial use.", "contradicts"),
    ("Writes to the queue are idempotent.", "Writes to the queue are not idempotent; duplicates appear on retry.", "contradicts"),
    ("The mobile client works offline for all screens.", "The mobile client fails offline on every screen.", "contradicts"),
    ("Index rebuilds are fast, under one minute for 1M rows.", "Index rebuilds are slow, over twenty minutes for 1M rows.", "contradicts"),
    ("The vendor SDK supports Android 8 and up.", "The vendor SDK does not support Android 8; it needs Android 10.", "contradicts"),
    # supports: newer confirms/backs older without replacing it
    ("Batching requests reduces the cost of the embedding service.", "A benchmark confirms that batching requests cut embedding service cost by 40 percent.", "supports"),
    ("The flaky test fails only when the clock is mocked.", "Re-running the flaky test with a real clock never failed in 50 runs, which backs the mocked-clock explanation.", "supports"),
    ("Users prefer short onboarding with three screens.", "Survey results from 200 users show most preferred a three-screen onboarding.", "supports"),
    ("The memory leak comes from the image cache.", "A heap profile shows the image cache holding 1.2 GB, consistent with the leak coming from it.", "supports"),
    ("SQLite is enough for the single-user store.", "Load tests with 100k records ran fine on SQLite, which supports using it for the single-user store.", "supports"),
    ("Pinning the compiler version avoids build drift.", "After pinning the compiler version, three weeks of builds were byte-identical, confirming that pinning avoids drift.", "supports"),
    ("The login failures are caused by expired tokens.", "Server logs show every failed login had an expired token, which confirms the cause.", "supports"),
    ("Compressing the uploads saves bandwidth.", "Measured upload traffic dropped by half after enabling compression, confirming the saving.", "supports"),
    # derived_from: newer is built from older (consequence, summary, refinement)
    ("Decision: the sync job must never delete remote files.", "Because the sync job must never delete remote files, the cleanup step is implemented as a dry-run report only.", "derived_from"),
    ("Spec: every record carries an id, a text and a date.", "Summary of the spec: records are triples of id, text and date.", "derived_from"),
    ("Meeting notes: we will ship the beta on the first of March, scope limited to search.", "Task list derived from the meeting notes: finish search, freeze other features, tag beta for March 1.", "derived_from"),
    ("The security review found three issues: weak hashing, open admin port, verbose errors.", "Fix plan built from the security review: rotate hashing to argon2, close the admin port, trim error messages.", "derived_from"),
    ("Policy: personal data never leaves the device.", "Following the policy that personal data never leaves the device, the analytics module only stores aggregate counts locally.", "derived_from"),
    ("Experiment log: variant B converted 4.1 percent against 3.2 percent for A.", "Writeup based on the experiment log: variant B lifted conversion from 3.2 to 4.1 percent.", "derived_from"),
    # none: same topic, none of the four relations clearly holds
    ("The deploy script runs on Node 18 in the staging pipeline.", "The deploy script prints its progress with coloured output.", "none"),
    ("Backups of the notes database run nightly.", "The notes database stores tags in a separate table.", "none"),
    ("The API rate limit for free users is 60 requests per minute.", "The API documentation lives in the docs folder and is written in Markdown.", "none"),
    ("Pull requests need one approval before merge.", "Pull request titles should start with a verb in the imperative mood.", "none"),
    ("The dashboard uses the blue theme by default.", "The dashboard footer shows the build number.", "none"),
    ("Logs are kept for 7 days in the ingestion service.", "The ingestion service listens on port 8081.", "none"),
]
