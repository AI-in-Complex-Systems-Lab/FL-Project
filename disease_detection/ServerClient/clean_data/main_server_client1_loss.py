import matplotlib.pyplot as plt

loss = [
    [0.23657500743865967, 0.22946204245090485, 0.2289057970046997, 0.22858601808547974, 0.22819030284881592],
    [0.22859445214271545, 0.22833053767681122, 0.22823265194892883, 0.22788089513778687, 0.2283162623643875],
    [0.22792212665081024, 0.2280280441045761, 0.2279994785785675, 0.2277735322713852, 0.227665975689888],
    [0.22772842645645142, 0.22786365449428558, 0.22775651514530182, 0.22755207121372223, 0.22780592739582062],
    [0.2275112271308899, 0.2275836169719696, 0.22776930034160614, 0.2278372198343277, 0.2276952862739563]
]


val_loss = [
    [0.2242426574230194, 0.22264625132083893, 0.22565701603889465, 0.2247891128063202, 0.22379276156425476],
    [0.22416284680366516, 0.2235846370458603, 0.22210700809955597, 0.222598135471344, 0.2228051722049713],
    [0.22421734035015106, 0.22377566993236542, 0.22504954040050507, 0.2248021364212036, 0.22368067502975464],
    [0.22337555885314941, 0.2234763205051422, 0.22376753389835358, 0.22332355380058289, 0.22495025396347046],
    [0.222159281373024, 0.2227424532175064, 0.22343432903289795, 0.22479590773582458, 0.22252358496189117]
]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

for i in range(5):
    plt.plot(loss[i], label=f'Round {i+1} Train Loss', marker = "o")

plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.xticks(range(len(accuracy[0])))


plt.subplot(2, 1, 2)
for i in range(5):
    plt.plot(val_loss[i], label=f'Round {i+1} Validation Loss', marker = "o")

plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.xticks(range(len(accuracy[0])))

plt.tight_layout()
plt.show()