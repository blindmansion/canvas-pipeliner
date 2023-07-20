import json
from collections import deque
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def llm_call(
    messages: List[Dict[str, str]], model="gpt-3.5-turbo", verbose=True
) -> Dict[str, str]:
    if verbose:
        print(f"Call messages: {messages}")

    # Call the OpenAI API
    completion = openai.ChatCompletion.create(model=model, messages=messages)

    response_content = completion.choices[0].message.content

    if verbose:
        print(f"LLM response: {response_content}")

    # Return only the response message
    return completion.choices[0].message


def has_cycle(graph):
    visited = set()
    rec_stack = set()

    for node in graph:
        if node not in visited:
            if dfs(node, visited, rec_stack, graph):
                return True
    return False


def dfs(node, visited, rec_stack, graph):
    visited.add(node)
    rec_stack.add(node)

    for neighbor in graph[node]:
        if neighbor not in visited:
            if dfs(neighbor, visited, rec_stack, graph):
                return True
        elif neighbor in rec_stack:
            return True

    rec_stack.remove(node)
    return False


def parse_canvas_file(file_path):
    """
    Parses a JSON file representing an Obsidian Canvas and returns various data structures representing groups, cards, links, labels, and colors.

    Args:
        file_path (str): The path to the JSON file to parse.

    Returns:
        tuple: A tuple containing dictionaries of groups, cards, links, group_labels, link_labels, card_colors, edge_colors, output_edges, and input_output_edges.
    """
    # Open the file and load the JSON data
    with open(file_path, "r") as file:
        canvas_data = json.load(file)

    # Initialize dictionaries and lists to store the parsed data
    groups = {}
    cards = {}
    links = []
    group_labels = {}
    link_labels = {}
    card_colors = {}
    edge_colors = {}
    output_edges = []
    input_output_edges = []
    card_spatial_data = {}

    # Define a mapping from color codes to color names
    color_map = {
        "1": "red",
        "2": "orange",
        "3": "yellow",
        "4": "green",
        "5": "cyan",
        "6": "purple",
    }

    # Loop over all nodes in the canvas
    for node in canvas_data["nodes"]:
        # If the node is a group, store its ID and label
        if node["type"] == "group":
            groups[node["id"]] = []
            group_labels[node["id"]] = node.get("label", "No Label")
        # If the node is a text (card), store its ID, text, position, and color
        elif node["type"] == "text":
            cards[node["id"]] = node["text"]
            card_spatial_data[node["id"]] = (node["x"], node["y"])
            if "color" in node:
                card_colors[node["id"]] = color_map.get(node["color"], "green")

    # Loop over all nodes again to assign cards to their respective groups
    for node in canvas_data["nodes"]:
        if node["type"] == "group":
            group_left = node["x"]
            group_right = node["x"] + node["width"]
            group_top = node["y"]
            group_bottom = node["y"] + node["height"]
            for card_id, card_text in cards.items():
                card_x, card_y = card_spatial_data[card_id]
                if (
                    group_left <= card_x <= group_right
                    and group_top <= card_y <= group_bottom
                ):
                    groups[node["id"]].append(card_id)

    # Loop over all edges in the canvas
    for edge in canvas_data["edges"]:
        # Store the color of the edge
        edge_colors[(edge["fromNode"], edge["toNode"])] = edge.get("color")

        # If the edge has a label, store it
        if "label" in edge:
            link_labels[(edge["fromNode"], edge["toNode"])] = edge["label"]

        # Add the edge to the links list
        links.append((edge["fromNode"], edge["toNode"]))

        # If the edge color is cyan or purple, add it to the respective list
        edge_color = edge.get("color")
        if edge_color == "5":  # Cyan
            output_edges.append((edge["fromNode"], edge["toNode"]))
        elif edge_color == "6":  # Purple
            input_output_edges.append((edge["fromNode"], edge["toNode"]))

    # Return all the parsed data
    return (
        groups,
        cards,
        links,
        group_labels,
        link_labels,
        card_colors,
        edge_colors,
        output_edges,
        input_output_edges,
    )


def create_messages_list(
    node, groups, cards, responses, replacements, order, group_replacements
):
    """
    Generate a list of messages for a given node (card) in the conversation.

    Parameters:
    node: The current node (card) being processed.
    groups: Dictionary of groups with their nodes.
    cards: Dictionary of cards with their content.
    responses: Dictionary of nodes with their responses from the language model.
    replacements: Dictionary of nodes with replacements to be made in their content.
    order: Dictionary indicating the order of processing for each node.
    group_replacements: Dictionary of replacements to be made in the current group's content.

    Returns:
    messages: List of message dictionaries generated for the current node.
    """
    # Identify the group to which the current node belongs
    group = next(
        (group for group, nodes in groups.items() if node in nodes), "no_group"
    )

    # Initialize a list to hold the messages
    messages = []

    # If the current node's group exists in the groups dictionary
    if group in groups:
        # Identify and sort the previous nodes in the group that have been processed
        previous_nodes = [n for n in groups[group] if n in order and n in responses]
        sorted_group = sorted(previous_nodes, key=lambda node: order[node])

        # For each of these previous nodes
        for previous_node in sorted_group:
            # Fetch the text of the card and perform replacements
            previous_message = cards[previous_node]
            for variable_name, replacement in group_replacements.items():
                placeholder = "{" + variable_name + "}"
                if placeholder in previous_message:
                    previous_message = previous_message.replace(
                        placeholder, replacement
                    )

            # Add a 'user' message with this text and the language model's response
            messages.append({"role": "user", "content": previous_message})
            messages.append(responses[previous_node])

    # Fetch the text of the current node, perform replacements, and add a 'user' message
    current_message = cards[node]
    for variable_name, replacement in group_replacements.items():
        placeholder = "{" + variable_name + "}"
        if placeholder in current_message:
            current_message = current_message.replace(placeholder, replacement)

    messages.append({"role": "user", "content": current_message})

    # Return the list of messages
    return messages


def process_outgoing_edges(
    node,
    outgoing_edges,
    labeled_links,
    responses,
    replacements,
    incoming_edges,
    queue,
    pending_replacements,
    edge_colors,
    output_edges,
    input_output_edges,
    cards,
    messages,
    group_replacements,
):
    """
    Process all outgoing edges from a given node in the conversation graph.

    Parameters:
    - node: The node from which outgoing edges are being processed.
    - outgoing_edges: A dictionary mapping each node to a list of its outgoing nodes.
    - labeled_links: A dictionary mapping each edge to its label, if it has one.
    - responses: A dictionary mapping each node to the response from the language model.
    - replacements: A dictionary mapping each node to a dictionary of variable replacements.
    - incoming_edges: A dictionary mapping each node to the count of its incoming edges.
    - queue: A queue of nodes to be processed.
    - pending_replacements: A dictionary mapping each node to a dictionary of pending variable replacements.
    - edge_colors: A dictionary mapping each edge to its color.
    - output_edges: A list of output edges.
    - input_output_edges: A list of input-output edges.
    - cards: A dictionary mapping each node to its content.
    - messages: A list of messages so far in the current group.
    - group_replacements: A dictionary of variable replacements for the current group.
    """
    for outgoing_node in outgoing_edges[node]:
        edge = (node, outgoing_node)

        # Check if the edge is an output edge
        if edge in output_edges:
            cards[outgoing_node] = f"<u>**OUTPUT**</u>\n\n{responses[node]['content']}"
            continue

        # Check if the edge is an input-output edge
        elif edge in input_output_edges:
            # Gather all messages in the group except the last one
            all_messages = [
                f"**{'USER' if m['role'] == 'user' else 'ASSISTANT'}:**\n{m['content']}"
                for m in messages[:-1]
            ]
            input_content = "\n\n".join(all_messages)
            cards[
                outgoing_node
            ] = f"<u>**INPUT**</u>\n\n{input_content}\n\n<u>**OUTPUT**</u>\n\n{responses[node]['content']}"
            continue

        # Check if the edge is labeled
        if edge in labeled_links:
            variable_name = labeled_links[edge]
            if node in responses:
                replacements[outgoing_node][variable_name] = responses[node]["content"]
                group_replacements[variable_name] = responses[node]["content"]
            else:
                pending_replacements[outgoing_node][variable_name] = node

        # Decrement the count of incoming edges to the outgoing node
        incoming_edges[outgoing_node] -= 1

        # If the outgoing node has no more incoming edges, add it to the queue
        if incoming_edges[outgoing_node] == 0:
            queue.append(outgoing_node)


def process_graph(
    groups: Dict[str, List[str]],
    cards: Dict[str, str],
    links: List[Tuple[str, str]],
    group_labels: Dict[str, str],
    link_labels: Dict[str, str],
    card_colors: Dict[str, str],
    edge_colors: Dict[Tuple[str, str], str],
    output_edges: List[Tuple[str, str]],
    input_output_edges: List[Tuple[str, str]],
    canvas: Dict,
) -> List[Dict[str, Any]]:
    """
    Process a graph that represents a conversation structure, make calls to a language model and handle the responses.

    Args:
    - groups: A dictionary mapping group IDs to lists of nodes in each group.
    - cards: A dictionary mapping node IDs to card text.
    - links: A list of tuples representing directed edges between nodes.
    - group_labels: A dictionary mapping group IDs to labels.
    - link_labels: A dictionary mapping edges to labels.
    - card_colors: A dictionary mapping node IDs to colors.
    - edge_colors: A dictionary mapping edges to colors.
    - output_edges: A list of edges representing 'Output' connections.
    - input_output_edges: A list of edges representing 'Input-Output' connections.
    - canvas: A dictionary representing the original Obsidian Canvas data.

    Returns:
    A dictionary mapping node IDs to responses from the language model.
    """

    # Initialize dictionaries to track incoming and outgoing edges of each node
    incoming_edges = {node: 0 for node in cards}
    outgoing_edges = {node: [] for node in cards}

    # Initialize a dictionary to track labeled links
    labeled_links = {}
    for link in links:
        from_node, to_node = link
        outgoing_edges[from_node].append(to_node)
        incoming_edges[to_node] += 1
        if link in link_labels:
            labeled_links[link] = link_labels[link]

    # Create a queue of nodes without incoming edges
    queue = deque(node for node, incoming in incoming_edges.items() if incoming == 0)

    # Initialize dictionaries to store the responses from the language model,
    # replacements for variable placeholders in card text,
    # and placeholders that still need to be replaced
    responses = {}
    replacements = {node: {} for node in cards}
    pending_replacements = {node: {} for node in cards}

    # Initialize a dictionary to track the processing order of nodes
    order = {}
    order_count = 0

    # Initialize a dictionary to store the replacements for each group
    group_replacements = {}

    # Process nodes in the queue
    while queue:
        node = queue.popleft()

        # Update the order dictionary and increment the order counter
        order[node] = order_count
        order_count += 1

        # Create a list of messages to be sent to the language model
        messages = create_messages_list(
            node, groups, cards, responses, replacements, order, group_replacements
        )

        # Call the language model and store the response
        response = llm_call(messages)
        responses[node] = response

        # Add the response to the messages after generating the formatted result
        messages.append(response)

        # Process outgoing edges from the current node
        process_outgoing_edges(
            node,
            outgoing_edges,
            labeled_links,
            responses,
            replacements,
            incoming_edges,
            queue,
            pending_replacements,
            edge_colors,
            output_edges,
            input_output_edges,
            cards,
            messages,
            group_replacements,
        )

    # Complete any remaining replacements in the card text
    for node, pending in pending_replacements.items():
        for variable_name, predecessor in pending.items():
            replacements[node][variable_name] = responses[predecessor]["content"]

    # Return the responses from the language model for each node
    return responses


def load_canvas_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_canvas_data(file_path, canvas_data):
    with open(file_path, "w") as file:
        json.dump(canvas_data, file, indent=2)


def update_canvas_data(canvas_data, cards):
    for node in canvas_data["nodes"]:
        if node["type"] == "text" and node["id"] in cards:
            node["text"] = cards[node["id"]]


def update_card_content(node, content, cards):
    # Update the content of the node in the cards dictionary
    cards[node] = content


def run_canvas_file(file_path: str):
    (
        groups,
        cards,
        links,
        group_labels,
        link_labels,
        card_colors,
        link_colors,
        output_edges,
        input_output_edges,
    ) = parse_canvas_file(file_path)

    # Build the graph
    graph = {node: [] for node in cards}
    for link in links:
        from_node, to_node = link
        graph[from_node].append(to_node)

    # Check for a cycle
    if has_cycle(graph):
        raise ValueError("The graph contains a cycle.")

    # If there's no cycle, continue processing the graph
    canvas_data = load_canvas_data(file_path)
    process_graph(
        groups,
        cards,
        links,
        group_labels,
        link_labels,
        card_colors,
        link_colors,
        output_edges,
        input_output_edges,
        canvas_data,
    )
    update_canvas_data(canvas_data, cards)
    save_canvas_data(file_path, canvas_data)


def main():
    run_canvas_file("/Users/nick/obsidian-vaults/omnit-testing-sync/color-test.canvas")


# make main function
if __name__ == "__main__":
    main()
