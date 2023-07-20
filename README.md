# Canvas Pipeliner

Use the obsidian canvas as a no-code llm pipeline editor

# Introduction

This application is designed to facilitate the creation of interactive conversational flows using the Obsidian Canvas plugin. By designing and running complex conversational scenarios with a language model, users can generate and store responses within the canvas.

## Cards and Connections

In the Obsidian Canvas format, 'cards' are text blocks containing a piece of conversation or instruction for the language model. 'Connections' define the order in which these cards are processed, guiding the flow of conversation. A connection from one card to another signifies that the content of the first card is processed before the second.

## Groups

Cards can be organized into 'groups', representing continuous chats between the user and the language model. The cards within a group are processed in the order defined by their connections.

## Interactions Between Groups or Ungrouped Nodes

For interactions between different groups or between ungrouped cards, connections must be labeled. The label on a connection becomes a variable in the target card. This variable should be wrapped in brackets (e.g., `{variable}`) in the card's content and will be replaced with the language model's response from the source card.

## Colors, Labels, and Special Cards

Colors and labels on connections carry special meanings and can create special types of cards:

- Cyan connections are 'Output' connections. The content of the target card is replaced by the response of the language model from the source card.
- Purple connections are 'Input-Output' connections. These create Input-Output cards, which display both the complete input sent to the language model and the output it generates. The input includes all the previous messages in the group, and the output is the language model's response to this input. This can be useful for reviewing the context in which the language model generated a particular response.

By automating the process of generating and storing these conversational flows, this application provides a powerful tool for creating complex, interactive scenarios with a language model.
