import matplotlib.pyplot as plt

# Provided data
data = {
    'Atelectasis': (0.8132371717673992, 0.39212266907321264),
    'Cardiomegaly': (0.9156352419510315, 0.31240260729808617),
    'Consolidation': (0.8086994968019762, 0.15280091662080167),
    'Edema': (0.9149406861173452, 0.22867592535108341),
    'Effusion': (0.8748722130764385, 0.5024526267152073),
    'Emphysema': (0.8925400126380519, 0.3193963060584463),
    'Fibrosis': (0.7964627586835628, 0.11395302166855598),
    'Hernia': (0.9786513313376398, 0.5686811861614125),
    'Infiltration': (0.7159173223429498, 0.3096100379776684),
    'Mass': (0.8283936921629478, 0.3049964355083898),
    'Nodule': (0.7797271567428682, 0.24904930348836993),
    'Pleural_Thickening': (0.8176153154310677, 0.155083463378881),
    'Pneumonia': (0.7610848251304815, 0.05138867278261005),
    'Pneumothorax': (0.8755341047057937, 0.36133136674414135)
}

# Extracting AUC and AU-PRC values
labels = list(data.keys())
auc_values = [value[0] for value in data.values()]
auprc_values = [value[1] for value in data.values()]

# Plotting AUC-ROC curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(labels, auc_values, marker='o', linestyle='-')
plt.title('AUC-ROC Curve')
plt.xlabel('Labels')
plt.ylabel('AUC Values')
plt.xticks(rotation=90)
plt.grid(True)

# Plotting AU-PRC curve
plt.subplot(1, 2, 2)
plt.plot(labels, auprc_values, marker='o', linestyle='-')
plt.title('AU-PRC Curve')
plt.xlabel('Labels')
plt.ylabel('AU-PRC Values')
plt.xticks(rotation=90)
plt.grid(True)

plt.tight_layout()
plt.show()
