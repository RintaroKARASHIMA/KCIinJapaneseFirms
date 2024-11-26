#%%
import pandas as pd
import numpy as np

#%%
original_colors_list = [
        '#ff0000',
        '#ff5000',
        '#ffa100',
        '#fff100',
        '#bbff00',
        '#6bff00',
        '#1aff00',
        '#00ff35',
        '#00ff86',
        '#00ffd6',
        '#00d6ff',
        '#0086ff',
        '#0035ff',
        '#1a00ff',
        '#6b00ff',
        '#bb00ff',
        '#ff00f1',
        '#ff00a1',
        '#ff0050',
        '#ff0000',
    ]

content_list = [
    ["Digital communication", "Electrical engineering", "24566", "100"],
    ["Telecommunications", "Electrical engineering", "34007", "96.97"],
    ["Computer technology", "Electrical engineering", "34246", "93.94"],
    ["Audio-visual technology", "Electrical engineering", "21662", "90.91"],
    ["IT methods for management", "Electrical engineering", "5565", "87.88"],
    ["Pharmaceuticals", "Chemistry", "62999", "84.85"],
    ["Organic fine chemistry", "Chemistry", "99687", "81.82"],
    ["Basic communication processes", "Electrical engineering", "6284", "78.79"],
    ["Optics", "Instruments", "17771", "75.76"],
    ["Semiconductors", "Electrical engineering", "17315", "72.73"],
    ["Biotechnology", "Instruments", "38361", "69.7"],
    ["Medical technology", "Electrical engineering", "46350", "66.67"],
    ["Micro-structural and nano-technology", "Chemistry", "2481", "63.64"],
    ["Measurement", "Instruments", "60591", "60.61"],
    ["Food chemistry", "Chemistry", "14635", "57.58"],
    ["Control", "Instruments", "17246", "54.55"],
    ["Furniture, games", "Other fields", "17389", "51.52"],
    ["Basic materials chemistry", "Chemistry", "43406", "48.48"],
    ["Chemical engineering", "Chemistry", "38982", "45.45"],
    ["Environmental technology", "Chemistry", "10611", "42.42"],
    ["Macromolecular chemistry, polymers", "Chemistry", "34069", "39.39"],
    ["Engines, pumps, turbines", "Mechanical engineering, machinery", "38953", "36.36"],
    ["Electrical machinery, apparatus, energy", "Electrical engineering", "57052", "33.33"],
    ["Textile and paper machines", "Mechanical engineering, machinery", "21638", "30.3"],
    ["Other consumer goods", "Other fields", "20302", "27.27"],
    ["Civil engineering", "Other fields", "32346", "24.24"],
    ["Materials, metallurgy", "Chemistry", "23955", "21.21"],
    ["Other special machines", "Mechanical engineering, machinery", "37208", "18.18"],
    ["Thermal processes and apparatus", "Mechanical engineering, machinery", "23146", "15.15"],
    ["Surface technology, coating", "Chemistry", "18977", "12.12"],
    ["Transport", "Mechanical engineering, machinery", "58920", "9.09"],
    ["Handling", "Mechanical engineering, machinery", "60120", "6.06"],
    ["Mechanical elements", "Mechanical engineering, machinery", "38604", "3.03"],
    ['Machine tools', 'Mechanical engineering, machinery', '0', '0'], 
    ['Analysis of biological materials', 'Instruments', '0', '0']
]

original_colors_list = [
        '#ff0000',
        '#ff5000',
        '#ffa100',
        '#fff100',
        '#bbff00',
        '#6bff00',
        '#1aff00',
        '#00ff35',
        '#00ff86',
        '#00ffd6',
        '#00d6ff',
        '#0086ff',
        '#0035ff',
        '#1a00ff',
        '#6b00ff',
        '#bb00ff',
        '#ff00f1',
        '#ff00a1',
        '#ff0050',
        '#ff0000',
    ]

set([content[1] for i, content in enumerate(content_list)])
tech_color = {
        'Chemistry': '#ff0000',
        'Electrical engineering': '#ffa100',
        'Instruments': '#1aff00', 
        'Mechanical engineering, machinery': '#0035ff',
        'Other fields': 'gray'
    }
        # '#ff5000',1
        # '#ffa100',2
        # '#fff100',3
        # '#bbff00',4
        # '#6bff00',5
        # '#1aff00',6
        # '#00ff35',7
        # '#00ff86',8
        # '#00ffd6',9
        # '#00d6ff',10
        # '#0086ff',11
        # '#0035ff',12
        # '#1a00ff',13
        # '#6b00ff',14
        # '#bb00ff',15
        # '#ff00f1',16
        # '#ff00a1',17
        # '#ff0050',18
        # '#ff0000',19

sample = {content[0]: tech_color[content[1]] for i, content in enumerate(content_list)}
sample

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_color(hex_color):
    """
    Display a color on a canvas based on the provided hexadecimal color code.

    Args:
    hex_color (str): The hexadecimal color code to display.

    Returns:
    None
    """
    # Create a figure and a subplot
    fig, ax = plt.subplots()
    # Create a rectangle patch with the given color and add it to the subplot
    rect = patches.Rectangle((0.1, 0.1), 0.6, 0.6, linewidth=1, edgecolor='r', facecolor=hex_color)
    ax.add_patch(rect)
    # Set limits and turn off axes for clarity
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Show the plot
    plt.text(0.5, 0.5, hex_color, ha='center', va='center', fontsize=12, color='black')
    plt.show()

# Example usage
for v in original_colors_list:
    show_color(v)
# show_color("#FF5733")


df = pd.DataFrame(content_list, columns=["schmoch35", "schmoch5", "reg_num_eu", "TCI_eu"])
df["period"] = "1985-2009"
df

df.to_csv('../../data/processed/external/abroad/eu.csv', 
          sep=',', 
          encoding='utf-8', 
          index=False)
