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
    with open(file_path, "r") as file:
        canvas_data = json.load(file)

    groups = {}
    cards = {}
    links = []
    group_labels = {}
    link_labels = {}
    card_colors = {}
    edge_colors = {}  # New dictionary to store the color of each edge
    output_edges = []  # New list to store the output edges
    input_output_edges = []  # New list to store the input-output edges

    card_spatial_data = {}

    color_map = {
        "1": "red",
        "2": "orange",
        "3": "yellow",
        "4": "green",
        "5": "cyan",
        "6": "purple",
    }

    for node in canvas_data["nodes"]:
        if node["type"] == "group":
            groups[node["id"]] = []
            group_labels[node["id"]] = node.get("label", "No Label")
        elif node["type"] == "text":
            cards[node["id"]] = node["text"]
            card_spatial_data[node["id"]] = (node["x"], node["y"])
            if "color" in node:
                card_colors[node["id"]] = color_map.get(node["color"], "green")

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

    for edge in canvas_data["edges"]:
        # Store the color of each edge
        color = edge.get("color")
        edge_colors[(edge["fromNode"], edge["toNode"])] = color

        if "label" in edge:
            link_labels[(edge["fromNode"], edge["toNode"])] = edge["label"]

        # Add the edge to the links list, regardless of its color
        links.append((edge["fromNode"], edge["toNode"]))

        # Check the color of the edge and store it accordingly
        edge_color = edge.get("color")
        if edge_color == "5":  # If the edge color is cyan
            output_edges.append((edge["fromNode"], edge["toNode"]))
        elif edge_color == "6":  # If the edge color is purple
            input_output_edges.append((edge["fromNode"], edge["toNode"]))

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


def create_messages_list(node, groups, cards, responses, replacements, order):
    group = next(
        (group for group, nodes in groups.items() if node in nodes), "no_group"
    )

    messages = []

    if group in groups:
        previous_nodes = [n for n in groups[group] if n in order and n in responses]
        sorted_group = sorted(previous_nodes, key=lambda node: order[node])
        for previous_node in sorted_group:
            messages.append({"role": "user", "content": cards[previous_node]})
            messages.append(responses[previous_node])

    # Create a new variable for the current card's message and perform replacements on it
    current_message = cards[node]
    for variable_name, replacement in replacements[node].items():
        placeholder = "{" + variable_name + "}"
        if placeholder in current_message:
            current_message = current_message.replace(placeholder, replacement)

    messages.append({"role": "user", "content": current_message})

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
):
    for outgoing_node in outgoing_edges[node]:
        edge = (node, outgoing_node)

        if edge in output_edges:
            cards[outgoing_node] = f"<u>**OUTPUT**</u>\n\n{responses[node]['content']}"
            continue
        elif edge in input_output_edges:
            # Gather all messages in the group except the last one
            all_messages = [
                f"**{'User' if m['role'] == 'user' else 'Assistant'}:** {m['content']}"
                for m in messages[:-1]
            ]
            input_content = "\n\n".join(
                all_messages
            )  # Concatenate all messages with newlines
            cards[
                outgoing_node
            ] = f"<u>**INPUT**</u>\n\n{input_content}\n\n<u>**OUTPUT**</u>\n\n{responses[node]['content']}"
            continue

        if edge in labeled_links:
            variable_name = labeled_links[edge]
            if node in responses:
                replacements[outgoing_node][variable_name] = responses[node]["content"]
            else:
                pending_replacements[outgoing_node][variable_name] = node

        incoming_edges[outgoing_node] -= 1

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
    incoming_edges = {node: 0 for node in cards}
    outgoing_edges = {node: [] for node in cards}

    labeled_links = {}
    for link in links:
        from_node, to_node = link
        outgoing_edges[from_node].append(to_node)
        incoming_edges[to_node] += 1
        if link in link_labels:
            labeled_links[link] = link_labels[link]

    queue = deque(node for node, incoming in incoming_edges.items() if incoming == 0)

    responses = {}
    replacements = {node: {} for node in cards}
    pending_replacements = {node: {} for node in cards}

    order = {}
    order_count = 0

    while queue:
        node = queue.popleft()

        order[node] = order_count
        order_count += 1

        messages = create_messages_list(
            node, groups, cards, responses, replacements, order
        )

        response = llm_call(messages)

        responses[node] = response

        messages.append(
            response
        )  # add the response to the messages after generating the formatted result

        # Process outgoing edges
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
            messages,  # Pass messages as a parameter
        )

    for node, pending in pending_replacements.items():
        for variable_name, predecessor in pending.items():
            replacements[node][variable_name] = responses[predecessor]["content"]

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
    run_canvas_file("/Users/nick/obsidian-vaults/omnit-testing-sync/simple3.canvas")


# make main function
if __name__ == "__main__":
    main()
