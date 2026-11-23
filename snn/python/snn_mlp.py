from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import torch
from sklearn.model_selection import train_test_split
from snntorch import surrogate
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(".\\2D_Geometric_Shapes_Dataset")

CLASS_FOLDERS = ["circle", "elipse", "fig_eight", "rectangle", "square", "triangle"]

IMG_H, IMG_W = 60, 80

T = 1

GRID_H = 12
GRID_W = 16
N_FEATURES = GRID_H * GRID_W * 2

SUBSET_SIZE = 1000000000
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
SEED = 42

SOBEL_PERCENTILE = 90
ALPHA = 3.0

SPIKE_PENALTY_WEIGHT = 1e-5
EARLY_EXIT_CONFIDENCE = 0.70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path(".\\snn_mlp_model")

MODELS_DIR = OUT_DIR / "models"
PLOTS_DIR = OUT_DIR / "plots"
REPORTS_DIR = OUT_DIR / "reports"

for d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


FEATURE_TYPE_NAME = "snn_mlp_on_off"
ARCHITECTURE_NAME = f"snn_mlp [T={T}, {N_FEATURES}] -> 64 -> 32 -> 4 with LIF3"


def image_to_spikes(img_u8):
    img_u8 = cv2.resize(
        img_u8,
        (IMG_W, IMG_H),
        interpolation=cv2.INTER_LINEAR,
    )

    img = img_u8.astype(np.float32) / 255.0

    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    sobel_combined = sobel_x + sobel_y

    pos_edge = np.maximum(sobel_combined, 0)
    neg_edge = np.maximum(-sobel_combined, 0)

    pos_edge /= pos_edge.max() + 1e-8
    neg_edge /= neg_edge.max() + 1e-8

    on_mask = pos_edge > np.percentile(pos_edge, SOBEL_PERCENTILE)
    off_mask = neg_edge > np.percentile(neg_edge, SOBEL_PERCENTILE)

    spikes = np.zeros((T, 2, IMG_H, IMG_W), dtype=np.float32)

    if T == 1:
        spikes[0, 0, on_mask] = 1.0
        spikes[0, 1, off_mask] = 1.0
    else:
        on_time = (np.exp(-ALPHA * pos_edge) * (T - 1)).astype(np.int32)
        off_time = (np.exp(-ALPHA * neg_edge) * (T - 1)).astype(np.int32)

        ys, xs = np.where(on_mask)
        spikes[on_time[ys, xs], 0, ys, xs] = 1.0

        ys, xs = np.where(off_mask)
        spikes[off_time[ys, xs], 1, ys, xs] = 1.0

    return spikes


def spikes_to_snn_mlp_features(spikes):
    T_steps, _, H, W = spikes.shape

    cell_h = H // GRID_H
    cell_w = W // GRID_W

    on = spikes[:, 0].reshape(T_steps, GRID_H, cell_h, GRID_W, cell_w)
    off = spikes[:, 1].reshape(T_steps, GRID_H, cell_h, GRID_W, cell_w)

    on_density = on.mean(axis=(2, 4))
    off_density = off.mean(axis=(2, 4))

    features = np.stack([on_density, off_density], axis=-1)
    features = features.reshape(T_steps, GRID_H * GRID_W * 2)

    return features.astype(np.float32)


class CableDataset(Dataset):
    def __init__(self, root, class_folders):
        self.samples = []

        for label, cname in enumerate(class_folders):
            folder = root / cname

            if not folder.exists():
                print("Missing folder:", folder)
                continue

            count = 0
            for p in folder.rglob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((p, label))
                    count += 1

            print(f"{cname}: label {label}, images={count}")

        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        spikes = image_to_spikes(img)
        features = spikes_to_snn_mlp_features(spikes)

        x = torch.from_numpy(features).float()
        y = torch.tensor(label, dtype=torch.long)

        return x, y


class CachedFeatureDataset(Dataset):
    def __init__(self, base_dataset, indices, name="dataset"):
        self.X = []
        self.Y = []
        self.samples = []

        print(f"\nPrecomputing {name} features into RAM...")

        for idx in tqdm(indices, desc=f"Caching {name} features", unit="img"):
            path, label = base_dataset.samples[idx]

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise RuntimeError(f"Could not read image: {path}")

            spikes = image_to_spikes(img)
            features = spikes_to_snn_mlp_features(spikes)

            self.X.append(features)
            self.Y.append(label)
            self.samples.append((path, label))

        self.X = np.asarray(self.X, dtype=np.float32)
        self.Y = np.asarray(self.Y, dtype=np.int64)

        print(f"{name} cached X shape:", self.X.shape)
        print(f"{name} cached Y shape:", self.Y.shape)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


class CableSNNMLP(nn.Module):
    def __init__(
        self,
        input_size=N_FEATURES,
        hidden1=64,
        hidden2=32,
        num_classes=4,
        beta=0.9,
    ):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(hidden2, num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(
        self,
        x,
        return_spike_stats=False,
        early_exit=False,
        confidence_threshold=0.90,
    ):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        logits_sum = 0
        steps_used = 0

        layer_spikes = {
            "layer1": torch.zeros(x.size(0), device=x.device),
            "layer2": torch.zeros(x.size(0), device=x.device),
            "layer3": torch.zeros(x.size(0), device=x.device),
        }

        T_steps = x.size(1)

        for t in range(T_steps):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Train using final membrane potential as class score
            logits_sum += mem3
            steps_used += 1

            layer_spikes["layer1"] += spk1.sum(dim=1)
            layer_spikes["layer2"] += spk2.sum(dim=1)
            layer_spikes["layer3"] += spk3.sum(dim=1)

            if early_exit:
                logits_avg_now = logits_sum / steps_used
                probs = torch.softmax(logits_avg_now, dim=1)
                max_conf = probs.max(dim=1).values

                if torch.all(max_conf >= confidence_threshold):
                    break

        logits_avg = logits_sum / steps_used

        if return_spike_stats:
            return logits_avg, layer_spikes, steps_used

        return logits_avg


def plot_accuracy(train_accs, val_accs, save_path=None):
    epochs = np.arange(1, len(train_accs) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accs,
        marker="o",
        linewidth=2,
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        val_accs,
        marker="o",
        linewidth=2,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close()


def export_fpga_weights(model, save_dir, scale=128):
    save_dir.mkdir(parents=True, exist_ok=True)

    def q_int16(arr):
        return np.round(arr * scale).astype(np.int16)

    layers = {
        "fc1_w": model.fc1.weight.detach().cpu().numpy(),
        "fc1_b": model.fc1.bias.detach().cpu().numpy(),
        "fc2_w": model.fc2.weight.detach().cpu().numpy(),
        "fc2_b": model.fc2.bias.detach().cpu().numpy(),
        "fc3_w": model.fc3.weight.detach().cpu().numpy(),
        "fc3_b": model.fc3.bias.detach().cpu().numpy(),
    }

    for name, arr in layers.items():
        np.save(save_dir / f"{name}_float.npy", arr)
        np.save(save_dir / f"{name}_int16.npy", q_int16(arr))

    header_path = save_dir / "fpga_weights.h"

    with open(header_path, "w") as f:
        f.write("#ifndef FPGA_WEIGHTS_H\n")
        f.write("#define FPGA_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define WEIGHT_SCALE {scale}\n")
        f.write(f"#define T_STEPS {T}\n")
        f.write(f"#define IMG_H {IMG_H}\n")
        f.write(f"#define IMG_W {IMG_W}\n")
        f.write(f"#define GRID_H {GRID_H}\n")
        f.write(f"#define GRID_W {GRID_W}\n")
        f.write(f"#define N_FEATURES {N_FEATURES}\n")
        f.write("#define HIDDEN1 64\n")
        f.write("#define HIDDEN2 32\n")
        f.write(f"#define NUM_CLASSES {len(CLASS_FOLDERS)}\n\n")

        for name, arr in layers.items():
            q = q_int16(arr)

            if q.ndim == 2:
                rows, cols = q.shape
                f.write(f"static const int16_t {name}[{rows}][{cols}] = {{\n")
                for r in range(rows):
                    row = ", ".join(str(int(v)) for v in q[r])
                    f.write(f"    {{{row}}},\n")
                f.write("};\n\n")

            elif q.ndim == 1:
                size = q.shape[0]
                values = ", ".join(str(int(v)) for v in q)
                f.write(f"static const int16_t {name}[{size}] = {{{values}}};\n\n")

        f.write("#endif\n")

    print("Saved FPGA weights to:", save_dir)
    print("Saved FPGA C header:", header_path)


def save_outputs(
    model,
    optimizer,
    train_accs,
    val_accs,
    train_losses,
    avg_val_steps_list,
):
    torch.save(
        model.state_dict(),
        MODELS_DIR / "snn_mlp_dict.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_folders": CLASS_FOLDERS,
            "feature_type": FEATURE_TYPE_NAME,
            "img_h": IMG_H,
            "img_w": IMG_W,
            "T": T,
            "grid_h": GRID_H,
            "grid_w": GRID_W,
            "n_features_per_timestep": N_FEATURES,
            "input_shape": [T, N_FEATURES],
            "architecture": ARCHITECTURE_NAME,
            "seed": SEED,
            "sobel_percentile": SOBEL_PERCENTILE,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "train_losses": train_losses,
            "avg_val_steps": avg_val_steps_list,
            "output_lif_layer": True,
            "loss_output": "final_lif3_membrane_potential",
            "spike_penalty_layers": ["lif1", "lif2", "lif3"],
        },
        MODELS_DIR / "snn_mlp_checkpoint.pt",
    )

    export_fpga_weights(
        model=model,
        save_dir=MODELS_DIR / "fpga_export",
        scale=128,
    )

    print("\nSaved outputs to:", OUT_DIR)


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset_full = CableDataset(ROOT, CLASS_FOLDERS)

    if len(dataset_full) == 0:
        raise RuntimeError("No samples found.")

    subset_size = min(SUBSET_SIZE, len(dataset_full))

    all_indices = np.arange(len(dataset_full))
    all_labels = np.array([label for _, label in dataset_full.samples])

    if subset_size == len(dataset_full):
        subset_indices = all_indices
    else:
        subset_indices, _ = train_test_split(
            all_indices,
            train_size=subset_size,
            stratify=all_labels,
            random_state=SEED,
        )

    subset_labels = all_labels[subset_indices]

    train_indices, val_indices = train_test_split(
        subset_indices,
        test_size=0.30,
        stratify=subset_labels,
        random_state=SEED,
    )

    print("Using subset:", len(subset_indices), "out of", len(dataset_full))
    print("Train samples:", len(train_indices))
    print("Val samples:", len(val_indices))

    train_set = CachedFeatureDataset(dataset_full, train_indices, name="train")
    val_set = CachedFeatureDataset(dataset_full, val_indices, name="val")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = CableSNNMLP(
        input_size=N_FEATURES,
        num_classes=len(CLASS_FOLDERS),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs = []
    train_losses = []
    avg_val_steps_list = []

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_l1_spikes = 0.0
        epoch_l2_spikes = 0.0
        epoch_l3_spikes = 0.0
        epoch_total_layer_spikes = 0.0
        epoch_samples = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits, layer_spikes, steps_used = model(
                x,
                return_spike_stats=True,
                early_exit=False,
            )

            ce_loss = criterion(logits, y)

            l1_spikes_per_image = layer_spikes["layer1"]
            l2_spikes_per_image = layer_spikes["layer2"]
            l3_spikes_per_image = layer_spikes["layer3"]

            total_layer_spikes_per_image = (
                l1_spikes_per_image + l2_spikes_per_image + l3_spikes_per_image
            )

            spike_penalty = SPIKE_PENALTY_WEIGHT * total_layer_spikes_per_image.mean()

            loss = ce_loss + spike_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += ce_loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

            batch_size = x.size(0)

            epoch_l1_spikes += l1_spikes_per_image.sum().item()
            epoch_l2_spikes += l2_spikes_per_image.sum().item()
            epoch_l3_spikes += l3_spikes_per_image.sum().item()
            epoch_total_layer_spikes += total_layer_spikes_per_image.sum().item()
            epoch_samples += batch_size

            if batch_idx % 100 == 0:
                input_nonzero = (x > 0).float().sum(dim=(1, 2)).mean().item()
                input_density = 100.0 * input_nonzero / (T * N_FEATURES)

                print(
                    f"Epoch {epoch + 1:02d}/{EPOCHS} "
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"ce_loss={ce_loss.item():.4f} | "
                    f"spike_penalty={spike_penalty.item():.6f} | "
                    f"loss={loss.item():.4f} | "
                    f"input_nonzero/img={input_nonzero:.1f}/{T * N_FEATURES} | "
                    f"input_density={input_density:.2f}% | "
                    f"L1/img={l1_spikes_per_image.mean().item():.1f} | "
                    f"L2/img={l2_spikes_per_image.mean().item():.1f} | "
                    f"L3/img={l3_spikes_per_image.mean().item():.1f}"
                )

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        avg_l1_spikes = epoch_l1_spikes / epoch_samples
        avg_l2_spikes = epoch_l2_spikes / epoch_samples
        avg_l3_spikes = epoch_l3_spikes / epoch_samples
        avg_total_layer_spikes = epoch_total_layer_spikes / epoch_samples

        model.eval()

        val_correct = 0
        val_total = 0
        val_steps_used = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                logits, _, steps_used = model(
                    x,
                    return_spike_stats=True,
                    early_exit=True,
                    confidence_threshold=EARLY_EXIT_CONFIDENCE,
                )

                pred = logits.argmax(dim=1)

                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                val_steps_used += steps_used * x.size(0)

        val_acc = val_correct / val_total
        avg_val_steps = val_steps_used / val_total

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        avg_val_steps_list.append(avg_val_steps)

        max_steps = T

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} DONE | "
            f"ce_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"avg_val_steps={avg_val_steps:.2f}/{max_steps} | "
            f"avg_L1/img={avg_l1_spikes:.1f} | "
            f"avg_L2/img={avg_l2_spikes:.1f} | "
            f"avg_L3/img={avg_l3_spikes:.1f} | "
            f"avg_total_layer/img={avg_total_layer_spikes:.1f}"
        )

    plot_accuracy(train_accs, val_accs, PLOTS_DIR / "accuracy_vs_epoch.png")

    save_outputs(
        model=model,
        optimizer=optimizer,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        avg_val_steps_list=avg_val_steps_list,
    )


if __name__ == "__main__":
    train()
