# Delta for Target Serialization

## ADDED Requirements

### Requirement: Plain Table Text Serialization

The target serialization system SHALL support table records whose canonical target contains plain recognized table text.

#### Scenario: Serialize UniRec table text

- GIVEN a canonical table record whose target contains `text: "项目\t金额\n收入\t100"`
- WHEN `table_text_v1` serializes the record
- THEN the serializer SHALL return the target text unchanged

#### Scenario: Reject missing table text

- GIVEN a canonical table record whose target lacks a `text` field
- WHEN `table_text_v1` serializes the record
- THEN the serializer SHALL fail with a clear error
