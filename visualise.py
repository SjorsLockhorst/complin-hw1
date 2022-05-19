from evaluate import eval_all

from prettytable import PrettyTable
import matplotlib.pyplot as plt

def visualise(show_table: bool = True, show_graph: bool = True, sample_size: int = 10):

    data = eval_all(verbose=False)
    all_data = []

    for ngram_type, rows in data.items():
        table = PrettyTable(["ngram", "limit", "sentence length", "correct %"])
        for row in rows:
            table.add_row(row)
            all_data.append(row)

        if show_table:
            print(ngram_type)
            print(table.get_string())

        with open(f"results/results_{ngram_type}.csv", "w") as f:
            f.write(table.get_csv_string())

    if show_graph:
        sorted_data = sorted(all_data, key=lambda x: x[0])
        selected_data = [row for row in sorted_data if row[2] == sample_size]
        xs = [row[0] for row in selected_data]
        ys = [row[3] for row in selected_data]

        plt.plot(xs, ys, label="Correct % for ngram of size N")

        plt.title(f"Ngram performance for test samples of size {sample_size}")
        plt.legend()
        plt.xlabel("N")
        plt.ylabel("Correct %")
        plt.savefig(f"./results/n_test_sample_{sample_size}")
        plt.show()

if __name__ == "__main__":
    sample_sizes = [10, 30, 90]
    for size in sample_sizes:
        visualise(sample_size=size)
