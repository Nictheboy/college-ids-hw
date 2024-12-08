# Convert .srt subtitles to CSV data file


import os
import csv


def srt_to_csv(srt_directory, output_csv, word_separator):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text"])

        filenames = os.listdir(srt_directory)
        filenames.sort()
        for filename in filenames:
            if filename.endswith(".srt"):
                with open(os.path.join(srt_directory, filename), "r", encoding="utf-8") as srt_file:
                    lines = srt_file.readlines()
                    text_lines = []
                    for line in lines:
                        line = line.strip()
                        if (
                            line
                            and not line.isdigit()
                            and "-->" not in line
                            and "\ufeff1" not in line
                        ):
                            text_lines.append(line)
                        else:
                            if text_lines:
                                writer.writerow(
                                    [word_separator.join(text_lines).replace("{\\an8}", "")]
                                )
                                text_lines = []
                    if text_lines:
                        writer.writerow([word_separator.join(text_lines).replace("{\\an8}", "")])


if __name__ == "__main__":
    srt_to_csv("data/chinese/srt", "data/chinese/all.csv", "")
    srt_to_csv("data/english/srt", "data/english/all.csv", " ")
