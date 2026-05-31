# Delta for Training Bootstrap

## ADDED Requirements

### Requirement: Preserve Literal Assistant Media Tokens

The SFT runtime dataset wrapper SHALL preserve literal media-token text in non-user messages without allowing it to be interpreted as a multimodal placeholder.

#### Scenario: Assistant text contains `<video>`

- GIVEN an SFT row whose user message contains an image placeholder
- AND whose assistant text contains the literal string `<video>`
- WHEN the runtime dataset builds VERL multi-turn messages
- THEN the user image placeholder SHALL still become an image segment
- AND the assistant message SHALL remain a text segment containing `<video>`

#### Scenario: Assistant text contains `<image>`

- GIVEN an SFT row whose assistant text contains the literal string `<image>`
- WHEN the runtime dataset builds VERL multi-turn messages
- THEN the assistant text SHALL remain text and SHALL NOT create an additional image segment
