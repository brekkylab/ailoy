# Translation Rules for Ailoy Documentation
This document is a guide for translating Ailoy documentation into other languages.


## Common Rules

### 1. Definitions 

- The original source of truth is English written under the path `docs/docs/`. 

- You can also refer the korean translation is written under the path `docs/i18n/ko/docusaurus-plugin-content-docs/current/` as qulified translation under the path `docs/i18n/ko`.

- You must generate the translation file under the path `docs/i18n/{language}/docusaurus-plugin-content-docs/current/` for each language.

- If the target language translation file already exists, you must update the translation file for the only new or updated content.

- If the target language translation file does not exist, you must create the translation file for the whole english content.


### 2. Code and Terminal Output 
- Code blocks (``` ```) and the contents of `<TerminalBox>` are **never translated** except for the comments.


### 3. Technical Terms

- For the technical term, if the users commonly use english term, you must use the english term or its Loanword in the target language.
- You must refer the other documents of the target language to check if you need to translate the technical term or not.
- Below examples are the wrong translations in Korean.

| English Term | Wrong translation in Korean  |
|-----------|-----------|
| Tool | Tool (도구 ❌) |
| Tool Description | Tool Description (Tool 설명 ❌) |
| Tool Behavior | Tool Behavior (Tool 동작 ❌) |
| Tool Call | Tool Call (Tool 호출 ❌) |
| Tool Result | Tool Result (Tool 결과 ❌) |

### Markdown Syntax Rules

1. `**[{not english text}](url)**` format is not rendered correctly in MDX.

| English Syntax | Wrong syntax in Korean  |
|-----------|-----------|
| **[Text](url)** | **[텍스트](url)** (❌) |


### 3.2 Bold + Parenthesis Pattern

`**{not english text}(english text)**` format is not rendered correctly in MDX. Itmust be rendered as `<strong>{english text}</strong>` in MDX.

```markdown
# Wrong example (Not rendered correctly)
**언어 모델(Language Model, LM)**
**채팅 완성(chat completion)**
**RAG(Retrieval-Augmented Generation)**

# Valid example (Rendered correctly)
<strong>언어 모델(Language Model, LM)</strong>
<strong>채팅 완성(chat completion)</strong>
<strong>RAG(Retrieval-Augmented Generation)</strong>
```