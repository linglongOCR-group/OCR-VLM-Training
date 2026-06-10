## ADDED Requirements

### Requirement: Runtime Reward Routing Profiles
The runtime VERL reward entrypoint SHALL support explicit reward routing profiles that select reward implementations by task without overloading `reward_version`.

#### Scenario: Route Formula to CDM and default to Levenshtein
- **GIVEN** runtime reward kwargs with `routing.default = normalized_levenshtein_v1` and `routing.by_task.formula = cdm_katex_v1`
- **WHEN** the VERL reward entrypoint scores a Formula sample
- **THEN** it SHALL select the `cdm_katex_v1` reward profile
- **AND** when it scores a non-Formula sample without a task-specific route, it SHALL select `normalized_levenshtein_v1`

#### Scenario: Reward version is provenance metadata
- **GIVEN** a runtime reward profile with `version = cdm_katex_v1`
- **WHEN** the reward entrypoint returns a result
- **THEN** the result SHALL include `reward_version = cdm_katex_v1`
- **AND** changing only the legacy top-level `reward_version` field SHALL NOT select a different reward implementation when explicit routing is absent

#### Scenario: Unknown reward profile fails clearly
- **GIVEN** runtime reward kwargs that route a task to an unconfigured reward profile ID
- **WHEN** the reward entrypoint attempts to score that task
- **THEN** it SHALL raise a clear configuration error naming the missing profile

#### Scenario: Unsupported reward type fails clearly
- **GIVEN** a configured reward profile whose `type` is not registered by the runtime reward registry
- **WHEN** the reward entrypoint attempts to build or use that profile
- **THEN** it SHALL raise a clear configuration error naming the unsupported reward type

### Requirement: Legacy Runtime Reward Compatibility
Existing Levenshtein-only GRPO reward configurations SHALL continue to work when no explicit routing profile is provided.

#### Scenario: Legacy reward kwargs use Levenshtein
- **GIVEN** runtime reward kwargs containing only `reward_version = levenshtein_v1`
- **WHEN** the VERL reward entrypoint scores any sample
- **THEN** it SHALL use the existing normalized Levenshtein reward behavior
- **AND** the returned result SHALL include `reward_version = levenshtein_v1`

#### Scenario: Legacy result includes score field
- **GIVEN** the legacy Levenshtein reward path returns `reward_total`
- **WHEN** the VERL reward entrypoint returns the result to VERL
- **THEN** it SHALL include `score` equal to `reward_total`
