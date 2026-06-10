## ADDED Requirements

### Requirement: CDM Service Preflight for GRPO Launch
The system SHALL perform CDM service preflight before launching a CDM-enabled GRPO run.

#### Scenario: CDM-enabled launch requires healthy service
- **GIVEN** a GRPO run configuration that routes any task to a CDM reward profile with preflight enabled
- **WHEN** launch preflight runs
- **THEN** it SHALL verify that the configured CDM service URL is reachable
- **AND** it SHALL verify that the service reports the expected version and browser readiness

#### Scenario: CDM scoring probe passes before launch
- **GIVEN** a reachable CDM service for a CDM-enabled GRPO run
- **WHEN** launch preflight runs
- **THEN** it SHALL score a small known Formula pair through the service
- **AND** it SHALL require the returned score to be numeric and in the range `[0.0, 1.0]`

#### Scenario: Failed CDM preflight stops launch
- **GIVEN** a CDM-enabled GRPO run whose CDM service preflight fails
- **WHEN** the launch command is evaluated
- **THEN** the system SHALL fail before starting training
- **AND** the error SHALL identify the service URL and the failed readiness condition

#### Scenario: Levenshtein-only launch does not require CDM service
- **GIVEN** a GRPO run configuration without any CDM reward route
- **WHEN** launch preflight runs
- **THEN** it SHALL NOT require the CDM service to be reachable
