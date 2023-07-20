# Canvas Pipeliner

Use the Obsidian canvas as a no-code LLM pipeline editor.

## Introduction

This application is designed to facilitate the creation of interactive conversational flows using the Obsidian Canvas plugin. By designing and running complex conversational scenarios with a language model, users can generate and store responses within the canvas.

## Cards and Connections

In the Obsidian Canvas format, 'cards' are text blocks containing a piece of conversation or instruction for the language model. 'Connections' define the order in which these cards are processed, guiding the flow of conversation. A connection from one card to another signifies that the content of the first card is processed before the second.

## Groups

Cards can be organized into 'groups', representing continuous chats between the user and the language model. The cards within a group are processed in the order defined by their connections.

## Variables in Card Content

For interactions between different groups or between ungrouped cards, connections must be labeled in order to pass text content. The label on a connection becomes a variable in the target card. This variable should be wrapped in brackets (e.g., `{variable}`) in the card's content and will be replaced with the language model's response from the source card.

## Linking to Obsidian Pages

Users can link to Obsidian pages within the card content using the `{[[Page Name]]}` syntax. The linked page's content will be extracted and displayed in place of the link, including the title of the page as `# Title`.

If a connection's label has an "@" symbol in front of it, like `@variable`, and the use of the variable in the card is also prefixed with "@", like `{@variable}`, the replacement will be treated as an Obsidian page name and the content of the corresponding page will be rendered in the target card. This allows the content of the pages to be dynamically used in the conversational flow. Both the variable name in the connection label and the use of the variable in the card must include the "@".

If the page name doesn't correspond to an existing page, the variable will be replaced with "The page x does not exist"

## Slash Commands in Connection Labels

In addition to colors, you can specify functionality using slash commands in connection labels. For example, if the first line in a connection label is a slash command like "/debug", the connection will be treated as a debug connection. This mirrors the functionality of a purple connection, but doesn't rely on color.

Current commands include:

- "/debug": This connection will create a Debug card, which displays both the complete input sent to the language model and the output it generates.
- "/output": This connection will create an Output card, with the content replaced by the response of the language model from the source card.

More commands will be added in future updates.

## Colors, Labels, and Special Cards

Colors and labels on connections, or slash commands in labels, carry special meanings and can create special types of cards:

- Cyan connections or those labeled with "/output" are 'Output' connections. The content of the target card is replaced by the response of the language model from the source card.
- Purple connections or those labeled with "/debug" are 'Debug' connections. These create Debug cards, which display both the complete input sent to the language model and the output it generates. The input includes all the previous messages in the group, and the output is the language model's response to this input. This can be useful for reviewing the context in which the language model generated a particular response.

By automating the process of generating and storing these conversational flows, this application provides a powerful tool for creating complex, interactive scenarios with a language model.
