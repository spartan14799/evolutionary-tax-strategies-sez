import os 

from pathlib import Path

import sys

import matplotlib.pyplot as plt

import networkx as nx 

import numpy as np 

from matplotlib.figure import Figure


import tkinter as tk

import ast 

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from typing import Dict, List, Tuple, Union




import pandas as pd 



# Initialize paths

# Set correct Working Directory 

# Path to project root (one level up from /tests)
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.simulation.economy.economy import Economy




accounts_path = ROOT_DIR /"chart_of_accounts.yaml"

# Static Dictionary with base agents info 

base_agents_info = { "MKT":
               {"type":"MKT",
                 "inventory_strategy": "FIFO", 
                 "firm_related_goods":[], 
                 "income_statement_type": "standard" ,
                "accounts_yaml_path": accounts_path, 
                "price_mapping":0} , 

                "NCT":
               {"type":"NCT",
                 "inventory_strategy": "FIFO", 
                 "firm_related_goods":[], 
                 "income_statement_type": "standard" ,
                "accounts_yaml_path": accounts_path, 
                "price_mapping":1} , 

                "ZF":
               {"type":"ZF",
                 "inventory_strategy": "FIFO", 
                 "firm_related_goods":[], 
                 "income_statement_type": "standard" ,
                "accounts_yaml_path": accounts_path, 
                "price_mapping":2}
               }


# Auxiliary Function to populate firm related goods assumming al agents can produce all goods in the production graph 



def populate_goods_agents_info(agents_info_dict , goods_list):


    for agent in agents_info_dict.keys():
        agents_info_dict[agent]['firm_related_goods'] = goods_list

    return agents_info_dict
       


def plot_graph(graph:nx.DiGraph):

    fig, ax = plt.subplots(figsize=(4,3))

    ax.clear()
    nx.draw(graph , with_labels = True , node_color = 'lightblue', ax = ax )

    return fig 
    


def run_model():

    # Links List information 
    raw_links = str(links_input.get('1.0',tk.END)) #! First Input 

    graph_links = ast.literal_eval(raw_links)
    
    print(graph_links)

    economy_graph = nx.DiGraph()

    economy_graph.add_edges_from(graph_links)

    related_goods = list(economy_graph.nodes()) 

    agents_dict = populate_goods_agents_info(base_agents_info,related_goods)

    price_matrix = np.array(ast.literal_eval(price_matrix_input.get('1.0',tk.END))) #! Second Input 

    economy_genome = np.array(ast.literal_eval(genome_input.get('1.0',tk.END))) #! Third Input 

    # Generate plot 

    fig = plot_graph(economy_graph)

    canvas = FigureCanvasTkAgg(fig,master = main_window)

    canvas.draw()

    canvas.get_tk_widget().grid(row = 3, column = 0 , padx = 10 , pady = 10)

    # Create Economy Object 

    economy = Economy(graph_links,price_matrix,agents_dict,economy_genome)

    production_plan = economy.orders

    results = economy.get_reports()

    report_NCT = results['reports']['NCT']

    report_ZF = results['reports']['ZF']

    nct_df = pd.DataFrame(list(report_NCT.items()), columns=['Account', 'Balance'])

    zf_df = pd.DataFrame(list(report_ZF.items()), columns=['Account', 'Balance'])



    utility = results['utility']

    # NCT Update 
    nct_text.delete("1.0", tk.END)

    nct_text.insert("1.0", nct_df.to_string(index=False))


    # ZF Update
    zf_text.delete("1.0", tk.END)
    zf_text.insert("1.0", zf_df.to_string(index=False))
  

     # Frame for production plan

    plan_df = pd.DataFrame(
    [(i, str(t)) for i, t in enumerate(production_plan, start=1)],
    columns=["Step", "Transaction"])

    # Plan Update 
    plan_text.delete('1.0', tk.END)
    plan_text.insert('1.0',plan_df.to_string(index=False))
   
    # Utility Update
    utility_text.delete("1.0", tk.END)
    utility_text.insert("1.0", str(utility))

    # Ledger Entries 

    open_aux_window()

    nct_ledger_entries = economy.agents['NCT'].accountant.ledger.get_all_entries()

    zf_ledger_entries = economy.agents['ZF'].accountant.ledger.get_all_entries()

    nct_entries_df = pd.DataFrame([(i, str(t)) for i, t in enumerate(nct_ledger_entries, start=1)],
    columns=["Step", "Entry"])

    zf_entries_df = pd.DataFrame([(i, str(t)) for i, t in enumerate(zf_ledger_entries, start=1)],
    columns=["Step", "Entry"])


    # Enter and Update Ledger 

    nct_ledger_text.delete('1.0', tk.END)
    nct_ledger_text.insert('1.0',nct_entries_df.to_string(index=False))

    zf_ledger_text.delete('1.0', tk.END)
    zf_ledger_text.insert('1.0',zf_entries_df.to_string(index=False))



 

# Generate GUI 
   
main_window = tk.Tk()

main_window.title('Economy Simulation Interface')

main_window.columnconfigure(0,weight = 1)

main_window.columnconfigure(1, weight = 2)

main_window.columnconfigure(2,weight = 1)

# Aux Window  - Ledger entries

aux_window = None 

aux_label = None 




def open_aux_window():
    global aux_window, aux_label, ledger_frame, nct_ledger_text, zf_ledger_text

    if aux_window is None or not tk.Toplevel.winfo_exists(aux_window):
        aux_window = tk.Toplevel(main_window)
        aux_window.title("Accounting Reports - Ledger Entry Lines")

        aux_label = tk.Label(aux_window, text="Ledger Reports")
        aux_label.pack(padx=10, pady=10)

        # Ledger frame belongs to aux_window
        ledger_frame = tk.Frame(aux_window)
        ledger_frame.pack(padx=10, pady=10)

        # NCT Ledger
        nct_ledger_label = tk.Label(ledger_frame, text="NCT Report - Entries", font=("Arial", 12, "bold"))
        nct_ledger_label.grid(row=0, column=0)

        nct_ledger_text = tk.Text(ledger_frame, height=30, width=120)
        nct_ledger_text.grid(row=1, column=0)

        # ZF Ledger
        zf_ledger_label = tk.Label(ledger_frame, text="ZF Report - Entries", font=("Arial", 12, "bold"))
        zf_ledger_label.grid(row=0, column=1)

        zf_ledger_text = tk.Text(ledger_frame, height=30,width=120)
        zf_ledger_text.grid(row=1, column=1)


# Inputs 

# Links input

links_label = tk.Label(main_window,text= 'Insert Graph Links list in list of tuples format').grid(row = 0 , column = 0)

links_input =  tk.Text(main_window , width = 40 , height = 5 , font = ('Arial' , 12))

links_input.grid(row = 1 , column = 0)


# Price Matrix Plot 

prices_label = tk.Label(main_window,text= 'Insert Price matrix in numpy array format').grid(row = 0 , column = 1)

price_matrix_input =  tk.Text(main_window , width = 40 , height = 20, font = ('Arial' , 12))

price_matrix_input.grid(row = 1 , column=1)


# Genome input 

genome_label = tk.Label(main_window,text= 'Insert Plan genome in Numpy Format').grid(row = 0 , column = 2)

genome_input =  tk.Text(main_window , width = 40 , height = 5,  font = ('Arial' , 12))

genome_input.grid(row = 1 , column = 2)


# Button 

run_button = tk.Button(main_window, text='Run plan with the Given Parameters', command=run_model)

run_button.grid(row = 4 , column=1)


### -------------------- The other Report Elements 

# Frame for reports
reports_frame = tk.Frame(main_window)
reports_frame.grid(row=2, column=1, padx=20, pady=10)

nct_label = tk.Label(reports_frame, text="NCT Report", font=("Arial", 12, "bold"))
nct_label.grid(row = 0, column = 0) 


nct_text = tk.Text(reports_frame, height=15, width=60)
nct_text.grid(row = 1, column = 0)


 # ZF Report
zf_label = tk.Label(reports_frame, text="ZF Report", font=("Arial", 12, "bold"))
zf_label.grid(row = 0, column = 1) 



zf_text = tk.Text(reports_frame, height=15, width=60)
zf_text.grid(row = 1, column = 1)

# Production Plan Report

plan_frame = tk.Frame(main_window)
plan_frame.grid(row=2, column=2, padx=10, pady=10)

plan_label = tk.Label(plan_frame, text="Production Plan", font=("Arial", 12, "bold"))
plan_label.grid(row=0, column=0)

plan_text = tk.Text(plan_frame, height=20, width=100)
plan_text.grid(row=1, column=0)

# Utility square (row 3, col 1)
utility_frame = tk.LabelFrame(main_window, text="Utility", font=("Arial", 12, "bold"))
utility_frame.grid(row=3, column=1, padx=10, pady=10)

utility_text = tk.Text(utility_frame, height=2, width=10)
utility_text.pack()


main_window.mainloop()





