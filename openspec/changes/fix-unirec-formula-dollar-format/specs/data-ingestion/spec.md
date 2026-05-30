## ADDED Requirements

### Requirement: UniRec40M Inline Formula Delimiters

The UniRec40M ingestion path SHALL normalize formulas embedded in text and table targets using MinerU-compatible unescaped `$...$` delimiters. It MUST NOT emit escaped delimiters such as `\$...\$` for inline formula annotations.

#### Scenario: Text record inline formula uses unescaped delimiters

- **GIVEN** a UniRec40M text label containing an inline formula source wrapper such as `\(x^2\)`
- **WHEN** the UniRec adapter exports the record as a canonical text task
- **THEN** the canonical text target SHALL contain `$x^2$`
- **AND** the canonical text target SHALL NOT contain `\$x^2\$`

#### Scenario: Table record inline formula uses unescaped delimiters

- **GIVEN** a UniRec40M table label containing an inline formula source wrapper such as `\(x^2\)`
- **WHEN** the UniRec adapter exports the record as a canonical table task
- **THEN** the canonical table target SHALL contain `$x^2$`
- **AND** the canonical table target SHALL NOT contain `\$x^2\$`

#### Scenario: Rebuilt view preserves unescaped delimiters

- **GIVEN** UniRec40M canonical text or table records whose targets contain inline formulas
- **WHEN** a UniRec40M SFT view is rebuilt from those canonical records
- **THEN** the assistant label SHALL preserve `$...$` inline formula delimiters
- **AND** the assistant label SHALL NOT contain `\$...\$` formula delimiters
