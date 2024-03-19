import matplotlib.pyplot as plt


def plot_normalized_bars(data):
    # Extracting and normalizing data
    groups = []
    values = []

    for item in data:
        try:
            params = item.parameters[0]
        except:
            continue
        groups.extend(params.keys())
        values.extend(params.values())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(groups, values, color="skyblue")
    plt.xlabel("Keywords")
    plt.ylabel("Normalized Percentage")
    plt.title("Normalized Keyword Distribution")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()
