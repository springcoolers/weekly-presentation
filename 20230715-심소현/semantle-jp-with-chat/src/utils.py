def format_result(word, sim, rank):
    return f"{word}: スコア {sim}, ランク {rank}"

def format_table(table, n_rows=10):
    top_results = table.sort_values(by="スコア", ascending=False).head(n_rows)
    return top_results.to_markdown(index=False)