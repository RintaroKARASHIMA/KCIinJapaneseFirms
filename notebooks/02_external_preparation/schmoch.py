# %%
import pandas as pd
import numpy as np
import re
from glob import glob
import regex
from googletrans import Translator


# %%
global data_dir, output_dir
data_dir = "../../data/original/external/schmoch/"
output_dir = "../../data/processed/external/schmoch/"


# %%
df_dict = pd.read_excel(
    f"{data_dir}ipc_technology.xlsx",
    sheet_name=None,
    engine="openpyxl",
    header=6,
    usecols=["Field_number", "Field_en", "IPC_code"],
)
df_dict.keys()

# %%
df_35 = df_dict["35 Fields of Technology"].copy()
df_35["IPC_code"] = df_35["IPC_code"].str.replace("%", "")
df_35

# %%
schmoch5_dict = {
    "Digital communication": "Electrical engineering",
    "Telecommunications": "Electrical engineering",
    "Computer technology": "Electrical engineering",
    "Audio-visual technology": "Electrical engineering",
    "IT methods for management": "Electrical engineering",
    "Pharmaceuticals": "Chemistry, pharmaceuticals",
    "Organic fine chemistry": "Chemistry, pharmaceuticals",
    "Basic communication processes": "Electrical engineering",
    "Optics": "Instruments",
    "Semiconductors": "Electrical engineering",
    "Biotechnology": "Instruments",
    "Medical technology": "Electrical engineering",
    "Micro-structural and nano-technology": "Chemistry, pharmaceuticals",
    "Measurement": "Instruments",
    "Food chemistry": "Chemistry, pharmaceuticals",
    "Control": "Instruments",
    "Furniture, games": "Other fields",
    "Basic materials chemistry": "Chemistry, pharmaceuticals",
    "Chemical engineering": "Chemistry, pharmaceuticals",
    "Environmental technology": "Chemistry, pharmaceuticals",
    "Macromolecular chemistry, polymers": "Chemistry, pharmaceuticals",
    "Engines, pumps, turbines": "Mechanical engineering, machinery",
    "Electrical machinery, apparatus, energy": "Electrical engineering",
    "Textile and paper machines": "Mechanical engineering, machinery",
    "Other consumer goods": "Other fields",
    "Civil engineering": "Other fields",
    "Materials, metallurgy": "Chemistry, pharmaceuticals",
    "Other special machines": "Mechanical engineering, machinery",
    "Thermal processes and apparatus": "Mechanical engineering, machinery",
    "Surface technology, coating": "Chemistry, pharmaceuticals",
    "Transport": "Mechanical engineering, machinery",
    "Handling": "Mechanical engineering, machinery",
    "Mechanical elements": "Mechanical engineering, machinery",
    "Machine tools": "Mechanical engineering, machinery",
    "Analysis of biological materials": "Instruments",
}

# %%
df_5 = df_35.copy()
df_5["Field_en"] = df_5["Field_en"].map(schmoch5_dict)
df_5
df_35["schmoch5"] = df_35["Field_en"].map(schmoch5_dict)
df_35

# %%
df_35.to_csv(f"{output_dir}35.csv", sep=",", encoding="utf-8", index=False)


# %%
description_str = """1. Electrical machinery, apparatus, energy: the field primarily covers the non-electronic part of electrical engineering, for instance, the generation, conversion and distribution of electric power, electric machines but also basic electric elements such as resistors, magnets, capacitors, lamps or cables. This field is often associated with “traditional” electrical engineering, but the high patent activity shows that technological innovation is still very important.
2. Audio-visual technology: audio-visual technology is largely equivalent to consumer electronics. The relevant IPC codes primarily refer to technologies and only sometimes products are directly addressed (H04R Loudspeakers ..., H04S Stereophonic systems)
3. Telecommunications: telecommunications is a very broad field covering a variety of techniques and products. The IPC codes are often quite technology-oriented, so that it is difficult to separate relevant product/applications areas such as mobile communication in a clear-cut field.
4. Digital communication: in the ISI-OST-INPI classification, this field was part of telecommunications. At present, it is a self-contained technology at the border between telecommunications and computer technology. A core application of this technology is the internet.
5. Basic communication processes: in the ISI-OST-INPI classification, this field was part of telecommunications. It covers very basic technologies such as oscillation, modulation, resonant circuits, impulse technique, coding/decoding. These techniques are used in telecommunications, computer technology, measurement, control. However, the explicit link to these fields by multiple classification is moderate, in the case of telecommunications 2.4 percent. So the definition as a separate field is justified. However, with 0.9 percent of all applications in 2005, it is the smallest fields of the present version of the classification.
6. Computer technology: this field is the largest of the proposed classification with 6.4 percent of all applications in 2005. Its size is already reduced by extracting field 7. The core area of C06F (Electrical digital processing) is defined in a very technical way (Arrangement for programme control, methods and arrangements for data conversion ...), so that a further break-down is difficult. It may be possible to separate specific application fields such as image data processing, recognition of data or speech analysis, but then these special fields may become too small.
7. IT methods for management: a major improvement of IPC8 is the introduction of the sub-class G06Q “Data processing methods, specially adapted for administrative, commercial, financial, managerial, supervisory or forecasting purposes”. This field represents software for these special purposes. In most countries, business methods are not patentable, but if they are admitted, they are registered in this sub-class. In any case, the size of this field is relevant with 1.2 percent of all applications in 2005. A combination of the fields 3 to 7 represents information technology in general. As the overlap is limited, this can be done by simple addition. The correct way is to combine the fields without double counting (unit)
8. Semiconductors: the field comprises semiconductors including methods for their production. Integrated circuits or photovoltaic elements belong to this field. The field includes microstructural technology (B81), as the number of applications in this sub-field is too small for a separate field.
9. Optics: this field covers all parts of traditional optical elements and apparatus, but also laser beam sources. In recent years new optical technologies such as optical switching have become more relevant.
10. Measurement: this field covers a broad variety of different techniques and applications. It would be possible to differentiate special sub-fields such as measuring of mechanical properties (length, oscillation, speed ...), but these sub-fields are generally too small.
11. Analysis of biological materials: this is the largest sub-field of “measurement” and was defined as a separate field. It primarily refers to the analysis of blood for medical purposes. In many cases, biotechnological methods are addressed.
12. Control: In the ISI-OST-INPI classification, this field was part of measuring & control. In recent years the part of control has become quantitatively more important, so that an independent field is justified. The field covers elements for controlling and regulating electrical and non-electrical systems and referring test arrangements, traffic control or signalling systems etc.
13. Medical technology: Medical technology is generally associated with high technology. However, a large part of the class A61 refers to less sophisticated products and technologies such as operating tables, massage devices, bandages etc. These less complex sub-fields represent a large number of patent applications, and the total field is the second largest of the suggested classification with 6.3 percent of all applications in 2005.
14. Organic fine chemistry: without further limitations, the applications in organic chemistry primarily refer to pharmaceuticals. More than 40 percent of the applications have an additional code in pharmaceuticals. As such a large overlap of fields is less appropriate for a classification system, all documents with co-classification in A61K were excluded. The major exception is the group A61K-008, which refers to cosmetics.
15. Biotechnology: biotechnology is defined as a separate field, although it is linked to a variety of different applications. Like organic chemistry or computer technology, it is a crosscutting or generic technology. However, the overlap with pharmaceuticals is too large, with a share of nearly 30 percent. Therefore, as in organic chemistry, applications with explicit co-classification in A61K are excluded.
16. Pharmaceuticals: this field refers to an area of application, not a technology. However, the key sub-class A61K is primarily organized by technologies (e.g., medicinal preparations containing inorganic active ingredients …). Cosmetics are explicitly excluded from the field; these represent about 10 percent of all applications classified in A61K.
17. Macromolecular chemistry, polymers: this field contains the chemical aspects of polymers. Machines for producing articles from plastics are classified in B29 and not included.
18. Food chemistry: this field represents 1.3 percent of the applications in 2005 and is one of the smallest fields in this classification. However, the growth of this field is remarkable, so that a higher weight can be assumed for the next years. Machines for food production are not included, but classified as part of field 28 (other special machines)
19. Basic materials chemistry: This field primarily covers typical mass chemicals such as herbicides, fertilisers, paints, petroleum, gas, detergents etc.
20. Materials, metallurgy: This field covers all types of metals, ceramics, glass or processes for the manufacture of steel.
21. Surface technology, coating: The coating of metals, generally with advanced methods represents the core of this field (C23). Furthermore it covers electrolytic processes, crystal growth and apparatus for applying liquids to surfaces. This field may be qualified as the high-tech part of field 20.
22. Micro-structure and nano-technology: This field covers micro-structural devices or systems, including at least one essential element or formation characterised by its very small size. It includes nano-structures having specialised features directly related to their size.
23. Chemical engineering: This field covers technologies at the borderline of chemistry and engineering. It refers to apparatus and processes for the industrial production of chemicals. Some of these processes may be classified as physical ones.
24. Environmental technology: This field covers a variety of different technologies and applications, in particular filters, waste disposal, water cleaning (a quite large area), gas-flow silencers and exhaust apparatus, waste combustion or noise absorption walls. However, it is not possible to define measuring of environmental pollution by IPC codes in a comprehensive way.
25. Handling: This field comprises elevators, cranes or robots, but also packaging devices. So in terms of research intensity, the field is quite heterogeneous.
26. Machine tools: The field is dominated by patent applications referring to turning, boring, grinding, soldering or cutting with a focus on metals.
27. Engines, pumps, turbines: This field covers non-electrical engines for all types of applications. In quantitative terms, applications for automobiles dominate.
28. Textile and paper machines: The fields 27 and 28 cover machines for specific production purposes. Textile and food machines represent the most relevant part of these machines and are classified separately.
29. Other special machines: see field 26.
30. Thermal processes and apparatus: The field covers applications such as steam generation, combustion, heating, refrigeration, cooling or heat exchange.
31. Mechanical elements: The field covers fluid-circuit elements, joints, shafts, couplings, valves, pipe-line systems or mechanical control devices. The focus is on engineering elements of machines such as joints or couplings.
32. Transport: the field covers all types of transport technology and applications with dominance of automotive technology. In principle, a separation of rail traffic and air traffic would be feasible, but the associated fields would be too small. In both cases, this is due to a low propensity to patent. The samples are quite small and not representative of the total technological activities in these sub-fields.
33. Furniture, games: this field represents the main parts of consumer goods in terms of the number of patent applications. The other consumer goods are a mix of many different technologies, all of them with low quantitative weight. Therefore a further differentiation is not useful. Even furniture and games combined comprise not more than 2.3 percent of all applications in 2005.
34. Other consumer goods: this field primarily represents less research-intensive sub-fields.
35. Civil engineering: the field covers construction of roads and buildings as well as elements of buildings such as locks, plumbing installations or strongrooms for valuables. A special part refers to mining which may be important for some countries. In general, the importance of this field is so low that the definition of a separate field is not justified.
"""
description_list = description_str.split("\n")
description_dict = {
    "field": [field.split(":")[0].split(". ")[-1] for field in description_list][:-1],
    "description": [description.split(":")[-1][1:] for description in description_list][
        :-1
    ],
}
description_en_df = pd.DataFrame(description_dict)
description_en_df.index += 1
description_en_df.to_csv(
    f"{output_dir}35_description_en.csv", sep=",", encoding="utf-8", index=False
)

# %%
description_ja_df = description_en_df.copy()
translator = Translator()
for i, row in description_ja_df.iterrows():
    description_ja_df.loc[i, "field"] = translator.translate(
        text=row["field"], dest="ja"
    ).text
    description_ja_df.loc[i, "description"] = translator.translate(
        text=row["description"], dest="ja"
    ).text
description_ja_df.to_csv(
    f"{output_dir}35_description_ja.csv", sep=",", encoding="utf-8", index=False
)
# %%
description_df = pd.concat(
    [
        description_en_df.rename(
            columns={"field": "field_en", "description": "description_en"}
        ),
        description_ja_df.rename(
            columns={"field": "field_ja", "description": "description_ja"}
        ),
    ],
    axis="columns",
)[["field_en", "field_ja", "description_en", "description_ja"]]
description_df.to_csv(
    f"{output_dir}35_description.csv", sep=",", encoding="utf-8", index=False
)


# %%
# energyを細かく見る場合
df = pd.concat(
    [value for value in df_dict.values()], axis="index", ignore_index=True
).drop_duplicates(subset=["IPC_code"], keep="first")
df["IPC_code"] = df["IPC_code"].str.replace("%", "").str.replace("#", "")
df["IPC_class"] = df["IPC_code"].str.replace("%", "").str.replace("#", "").str[:3]
df["IPC_subclass"] = df["IPC_code"].str.replace("%", "").str.replace("#", "").str[:4]
df
df[(df["IPC_code"] != df["IPC_subclass"]) & (df["IPC_code"] != df["IPC_class"])]

for uni_str in df[
    (df["IPC_code"] != df["IPC_subclass"]) & (df["IPC_code"] != df["IPC_class"])
]["IPC_subclass"].unique():
    print("=======================================")
    display(df[df["IPC_code"] == uni_str])
    display(df[df["IPC_subclass"] == uni_str])
