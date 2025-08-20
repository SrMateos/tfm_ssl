import logging
import os
import time

import matplotlib.pyplot as plt
import mlflow
import logging
import os
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss, PerceptualLoss, SSIMLoss
from monai.visualize import matshow3d
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from torch.nn import L1Loss
from tqdm import tqdm
import numpy as np
from zmq import EVENT_CLOSE_FAILED

from src.constants import ALL_TASK1, ALL_TASKS
from src.data_handling import (
    get_data_paths,
    get_vae_train_transforms,
    get_vae_val_transforms,
    get_vae_post_transforms,
)
from src.data_handling.datasets import split_data
from src.losses import (KLLoss, FFL3D)
from src.metrics.image_metrics import ImageMetrics
from src.utils.mlflow_utils import setup_mlflow, log_config
from monai.transforms.utils import allow_missing_keys_mode
from monai.data import decollate_batch

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.image_metrics = ImageMetrics()

        self._setup_logging()
        self._setup_mlflow()
        self._load_config_params()

        logging.info(f"Trainer initialized with config: {self.config}")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _setup_mlflow(self):
        experiment_name = self.config["mlflow"].get("experiment_name", "SSL_Training")
        print(f"Setting up MLflow experiment: {experiment_name}")

        self.run = setup_mlflow(self.config, experiment_name=experiment_name)
        log_config(self.config)
        mlflow.log_param("device", self.device.type)

    def _load_config_params(self):
        # Data settings
        self.patch_size = tuple(self.config["data"].get("patch_size", [64, 64, 64]))
        self.train_split = self.config["data"].get("train_split", 0.7)
        self.val_split = self.config["data"].get("val_split", 0.15)
        self.test_split = self.config["data"].get("test_split", 0.15)
        self.task1 = self.config["data"].get("task1", True)
        self.debug_mode = bool(self.config.get("debug", False))

        # Training settings
        self.batch_size = self.config["training"].get("batch_size", 1)
        self.num_epochs = self.config["training"].get("epochs", 200)
        self.learning_rate = float(self.config["training"].get("learning_rate", 1e-4))

        # Loss weights
        self.pixel_wise_weight = float(self.config["training"].get("pixel_wise_weight", 1.0))
        self.perceptual_weight = float(
            self.config["training"].get("perceptual_weight", 0.001)
        )
        self.ssim_weight = float(self.config["training"].get("ssim_weight", 0.1))
        self.focal_frequency_weight = float(self.config["training"].get("focal_frequency_weight", 1e-4))

        self.adv_weight = float(self.config["training"].get("adv_weight", 0.01))
        self.max_kl_weight = float(self.config["training"].get("max_kl_weight", 1e-2))
        self.kl_annealing_cycles = self.config["training"].get("kl_annealing_cycles", 4)
        self.kl_annealing_ratio = self.config["training"].get("kl_annealing_ratio", 0.5)

        self.warmup_epochs = self.config["training"].get("warmup_epochs", 5)
        self.val_interval = self.config["training"].get("val_interval", 5)
        self.save_interval = self.config["training"].get("save_interval", 20)

        # Inference settings
        self.sw_batch_size = self.config["inference"].get("sw_batch_size", 1)
        self.overlap = self.config["inference"].get("overlap", 0.5)
        self.mode = self.config["inference"].get("mode", "gaussian")

    def _log_hyperparameters(self):
        mlflow.log_params(
            {
                "patch_size": str(self.patch_size),
                "train_split": self.train_split,
                "val_split": self.val_split,
                "test_split": self.test_split,
                "debug_mode": self.debug_mode,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "pixel_wise_weight": self.pixel_wise_weight,
                "perceptual_weight": self.perceptual_weight,
                "ssim_weight": self.ssim_weight,
                "focal_frequency_weight": self.focal_frequency_weight,
                "max_kl_weight": self.max_kl_weight,
                "adv_weight": self.adv_weight,
                "warmup_epochs": self.warmup_epochs,
                "val_interval": self.val_interval,
                "save_interval": self.save_interval,
                "sw_batch_size": self.sw_batch_size,
                "overlap": self.overlap,
                "inference_mode": self.mode,
                "random_seed": 42,
            }
        )

    def _prepare_data(self):
        logging.info("Loading data...")
        _, cts_paths, masks_paths = get_data_paths(
            ALL_TASKS, task1=self.task1, debug=self.debug_mode
        )
        data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]
        logging.info(f"First data sample: {data[0]}")

        train_data, val_data, test_data = split_data(
            data=data,
            train_split=self.train_split,
            val_split=self.val_split,
            test_split=self.test_split,
            random_seed=42,
        )

        logging.info(
            f"Data split: {len(train_data)} train samples, {len(val_data)} validation samples"
        )

        train_ds = CacheDataset(
            data=train_data,
            transform=get_vae_train_transforms(patch_size=self.patch_size),
            cache_rate=0.2,
        )

        val_ds = CacheDataset(
            data=val_data,
            transform=get_vae_val_transforms(patch_size=self.patch_size),
            cache_rate=1,
        )

        test_ds = CacheDataset(
            data=test_data,
            transform=get_vae_val_transforms(patch_size=self.patch_size),
            cache_rate=1,
        )

        self.val_transforms = val_ds.transform

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

        mlflow.log_param("dataset_size", len(data))
        mlflow.log_param("train_dataset_size", len(train_data))
        mlflow.log_param("val_dataset_size", len(val_data))
        mlflow.log_param("test_dataset_size", len(test_data))

    def _setup_models(self):
        self.autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 64),
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, True),
        )
        self.autoencoder.to(self.device)
        mlflow.log_param("autoencoder_name", self.autoencoder.__class__.__name__)
        mlflow.log_param("autoencoder_architecture_summary", str(self.autoencoder))

        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            channels=32,
            in_channels=1,  # autoencoder input
            out_channels=1,
            norm="INSTANCE"
        )
        self.discriminator.to(self.device)
        mlflow.log_param("discriminator_name", self.discriminator.__class__.__name__)
        mlflow.log_param("discriminator_architecture_summary", str(self.discriminator))

    def _setup_losses(self):
        self.l1_loss = L1Loss().to(self.device)
        self.loss_perceptual = PerceptualLoss(
            spatial_dims=3,
            network_type="radimagenet_resnet50",
            is_fake_3d=True,
            fake_3d_ratio=0.25,
        ).to(self.device)
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=7).to(self.device)
        self.focal_frequency_loss = FFL3D().to(self.device)
        self.adv_loss = PatchAdversarialLoss(criterion="hinge").to(self.device)
        self.adv_loss.real_label = 0.9
        self.kl_loss = KLLoss().to(self.device)

    def _setup_optimizers(self):
        self.optimizer_g = torch.optim.Adam(
            params=self.autoencoder.parameters(), lr=self.learning_rate
        )
        self.optimizer_d = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=5e-5
        )
        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_g,
            mode="min",
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            threshold=1e-3,
        )

    def _train_generator_step(self, images, epoch):
        self.optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = self.autoencoder(images)

        if self.ssim_weight > 0:
            reconstruction = reconstruction.clamp(0, 1)
            loss_ssim = self.ssim_loss(reconstruction, images)
        if self.pixel_wise_weight > 0:
            loss_recon = self.l1_loss(reconstruction, images)
        if self.perceptual_weight > 0:
            loss_perc = self.loss_perceptual(reconstruction.float(), images.float())
        if self.focal_frequency_weight > 0:
            loss_focal = self.focal_frequency_loss(reconstruction, images)

        kl_loss = self.kl_loss(z_mu, z_sigma)

        loss_g_base = (
            (self.pixel_wise_weight * loss_recon + self.ssim_weight * loss_ssim
                + self.focal_frequency_weight * loss_focal + self.perceptual_weight * loss_perc)
                + self.kl_weight * kl_loss
        )

        loss_g_adv = 0.0
        if epoch >= self.warmup_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            loss_g_adv = self.adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss_g = loss_g_base + self.adv_weight * loss_g_adv
        else:
            loss_g = loss_g_base

        loss_g.backward()
        self.optimizer_g.step()

        ret_dictionary = {
            "total": loss_g.item(),
            "kl": kl_loss.item(),
            "adversarial": (
                loss_g_adv.item()
                if isinstance(loss_g_adv, torch.Tensor)
                else loss_g_adv
            ),
            "reconstruction_tensor": reconstruction,
        }

        if self.pixel_wise_weight > 0:
            ret_dictionary["pixel-wise"] = loss_recon.item()
        if self.perceptual_weight > 0:
            ret_dictionary["perceptual"] = loss_perc.item()
        if self.ssim_weight > 0:
            ret_dictionary["ssim"] = loss_ssim.item()
        if self.focal_frequency_weight > 0:
            ret_dictionary["focal_frequency"] = loss_focal.item()

        return ret_dictionary

    def _train_discriminator_step(self, reconstruction, images, epoch):
        if epoch < self.warmup_epochs:
            return {"total": 0.0, "fake": 0.0, "real": 0.0}

        self.optimizer_d.zero_grad(set_to_none=True)
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(
            logits_fake, target_is_real=False, for_discriminator=True
        )

        logits_real = self.discriminator(images.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(
            logits_real, target_is_real=True, for_discriminator=True
        )

        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
        loss_d = self.adv_weight * discriminator_loss

        loss_d.backward()
        self.optimizer_d.step()

        return {
            "total": loss_d.item(),
            "fake": loss_d_fake.item(),
            "real": loss_d_real.item(),
        }

    def _validate_epoch(self, epoch):
        self.autoencoder.eval()
        val_step = 0
        val_recon_loss = 0
        val_metrics = {"mae": [], "psnr": [], "ms_ssim": []}
        val_bar = tqdm(
            total=len(self.val_loader), desc="Validation", unit="step", leave=False
        )

        post_tf = get_vae_post_transforms(self.val_transforms)

        with torch.no_grad():
            for val_batch in self.val_loader:
                val_step += 1
                val_images = val_batch["image"].to(self.device)

                val_outputs = sliding_window_inference(
                    inputs=val_images,
                    roi_size=self.patch_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.autoencoder.reconstruct,
                    overlap=self.overlap,
                    mode=self.mode,
                    device=self.device,
                    sw_device=self.device,
                )

                if self.config["mlflow"].get("log_images", False) and val_step == 1:
                    self._log_validation_images(val_images, val_outputs, epoch)
                    self._log_validation_volume(val_outputs, epoch)

                batch_recon_loss = self.l1_loss(val_outputs, val_images)
                val_recon_loss += batch_recon_loss.item()

                val_batch["pred"] = val_outputs

                val_samples = decollate_batch(val_batch)

                with allow_missing_keys_mode(self.val_transforms):
                    restored = [post_tf(val_sample) for val_sample in val_samples]

                print (f"restored keys: {restored[0].keys()}")
                print(f"type of restored: {type(restored)}")
                print(f"length of restored: {len(restored)}")
                print(f"Max value in restored: {restored[0]['image'].max()} - Min value: {restored[0]['image'].min()}")
                print(f"Max value in pred: {restored[0]['pred'].max()} - Min value: {restored[0]['pred'].min()}")

                gt = np.squeeze(restored[0]["image"].cpu().numpy())
                recon = np.squeeze(restored[0]["pred"].cpu().numpy())
                mask = np.squeeze(restored[0]["mask"].cpu().numpy())

                scores = self.image_metrics.score_patient(
                    gt_img=gt, synthetic_ct=recon, mask=mask
                )

                tqdm.write(
                    f"Validation Step {val_step} - MAE: {scores['mae']:.4f}, "
                    f"PSNR: {scores['psnr']:.4f}, MS-SSIM: {scores['ms_ssim']:.4f}"
                )

                for key, value in scores.items():
                    val_metrics[key].append(value)

                val_bar.update(1)

        val_bar.close()
        avg_val_recon_loss = val_recon_loss / val_step
        avg_metrics = {f"val_{k}": np.mean(v) for k, v in val_metrics.items()}

        mlflow.log_metrics({"val_recon_loss": avg_val_recon_loss}, step=epoch)
        mlflow.log_metrics(avg_metrics, step=epoch)

        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        tqdm.write(
            f"  Validation - Avg Recon Loss: {avg_val_recon_loss:.4f}, {metrics_str}"
        )

        return avg_val_recon_loss

    def _test_epoch(self):
        self.autoencoder.eval()
        test_step = 0
        test_recon_loss = 0
        test_metrics = {"mae": [], "psnr": [], "ms_ssim": []}
        test_bar = tqdm(
            total=len(self.test_loader), desc="Testing", unit="step", leave=False
        )

        post_tf = get_vae_post_transforms(self.val_transforms)

        with torch.no_grad():
            for test_batch in self.test_loader:
                test_step += 1
                test_images = test_batch["image"].to(self.device)

                test_outputs = sliding_window_inference(
                    inputs=test_images,
                    roi_size=self.patch_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.autoencoder.reconstruct,
                    overlap=self.overlap,
                    mode=self.mode,
                    device=self.device,
                    sw_device=self.device,
                )

                if self.config["mlflow"].get("log_images", False):
                    self._log_validation_volume(test_images, epoch=test_step, test=True)
                    self._log_validation_volume(test_outputs, epoch=test_step, test=True)

                batch_recon_loss = self.l1_loss(test_outputs, test_images)
                test_recon_loss += batch_recon_loss.item()

                test_batch["pred"] = test_outputs

                test_samples = decollate_batch(test_batch)

                with allow_missing_keys_mode(self.val_transforms):
                    restored = [post_tf(test_sample) for test_sample in test_samples]

                gt = np.squeeze(restored[0]["image"].cpu().numpy())
                recon = np.squeeze(restored[0]["pred"].cpu().numpy())
                mask = np.squeeze(restored[0]["mask"].cpu().numpy())

                scores = self.image_metrics.score_patient(
                    gt_img=gt, synthetic_ct=recon, mask=mask
                )

                tqdm.write(
                    f"Test Step {test_step} - MAE: {scores['mae']:.4f}, "
                    f"PSNR: {scores['psnr']:.4f}, MS-SSIM: {scores['ms_ssim']:.4f}"
                )

                for key, value in scores.items():
                    test_metrics[key].append(value)

                test_bar.update(1)

        test_bar.close()
        avg_test_recon_loss = test_recon_loss / test_step
        avg_metrics = {f"test_{k}": np.mean(v) for k, v in test_metrics.items()}

        mlflow.log_metrics({"test_recon_loss": avg_test_recon_loss}, step=0)
        mlflow.log_metrics(avg_metrics, step=0)

        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        tqdm.write(
            f"  Test - Avg Recon Loss: {avg_test_recon_loss:.4f}, {metrics_str}"
        )

        return avg_test_recon_loss, avg_metrics


    def _log_validation_images(self, val_images, val_outputs, epoch):
        img_slice = val_images[0, 0, :, :, val_images.shape[-1] // 2].cpu().numpy()
        recon_slice = (
            val_outputs[0, 0, :, :, val_outputs.shape[-1] // 2].cpu().numpy()
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title("Original Image Slice")
        axes[0].axis("off")
        axes[1].imshow(recon_slice, cmap="gray")
        axes[1].set_title("Reconstruction Slice")
        axes[1].axis("off")
        plt.tight_layout()
        mlflow.log_figure(fig, f"validation_image_epoch_{epoch+1}.png")
        plt.close(fig)

    def _log_validation_volume(self, val_outputs, epoch=None, test=False, test_original=False):
        if not test:
            title = "Validation Reconstruction Volume" if epoch is not None else "Original Volume"
            artifact_name = f"validation_volume_epoch_{epoch+1}.png" if epoch is not None else "original_volume.png"
        else:
            title = "Test Reconstruction Volume"
            artifact_name = f"test_reconstruction_volume_epoch_{epoch}.png" if not test_original else f"test_original_volume_epoch_{epoch}.png"

        fig, _ = matshow3d(
            volume=val_outputs[0].cpu().numpy(),
            title=title,
            figsize=(50, 50),
            every_n=10,
            frame_dim=-1,
            cmap="gray",
            show=False,
        )
        plt.tight_layout()
        mlflow.log_figure(fig, artifact_name)
        plt.close(fig)

    def _update_kl_weight(self, epoch):
        if self.kl_annealing_cycles == 0:
            self.kl_weight = self.max_kl_weight
            return

        period = self.num_epochs / self.kl_annealing_cycles
        epoch_in_cycle = epoch % period
        tau = epoch_in_cycle / (period * self.kl_annealing_ratio)

        if tau <= 1.0:
            self.kl_weight = float(self.max_kl_weight * tau)
        else:
            self.kl_weight = float(self.max_kl_weight)

    def _save_best_models(self, best_val_metric, epoch):
        mlflow.log_metric("best_val_recon_loss_epoch", epoch)
        mlflow.log_metric("best_val_recon_loss", best_val_metric)

        best_autoencoder_path = "best_autoencoder_model.pth"
        best_discriminator_path = "best_discriminator_model.pth"

        torch.save(self.autoencoder.state_dict(), best_autoencoder_path)
        torch.save(self.discriminator.state_dict(), best_discriminator_path)

        mlflow.log_artifact(best_autoencoder_path, artifact_path="models")
        mlflow.log_artifact(best_discriminator_path, artifact_path="models")

        tqdm.write(
            f"New best model saved with validation recon loss: {best_val_metric:.4f}"
        )

    def _save_checkpoint(self, epoch):
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        autoencoder_ckpt_path = os.path.join(
            checkpoint_dir, f"autoencoder_epoch_{epoch+1}.pth"
        )

        discriminator_ckpt_path = os.path.join(
            checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth"
        )

        torch.save(self.autoencoder.state_dict(), autoencoder_ckpt_path)
        torch.save(self.discriminator.state_dict(), discriminator_ckpt_path)

        mlflow.log_artifact(autoencoder_ckpt_path, artifact_path="checkpoints")
        mlflow.log_artifact(discriminator_ckpt_path, artifact_path="checkpoints")

        tqdm.write(f"Checkpoint saved at epoch {epoch+1}")

    def train(self):
        self._prepare_data()
        self._setup_models()
        self._setup_losses()
        self._setup_optimizers()

        # Log first original volume with self._log_validation_volume to compare with validation outputs
        if self.config["mlflow"].get("log_original_volume", False):
            original_volume = self.val_loader.dataset[0]["image"]
            self._log_validation_volume(original_volume, epoch=None)

        logging.info("Starting training...")

        best_val_metric = float("inf")
        self.kl_weight = 0.0  # Start with 0

        for epoch in range(self.num_epochs):
            self.autoencoder.train()
            self.discriminator.train()
            epoch_start = time.time()

            self._update_kl_weight(epoch)

            epoch_losses = {
                "g_loss": 0, "d_loss": 0, "pixel_wise": 0,
                "perceptual": 0, "ssim": 0, "focal_frequency": 0, "kl": 0,
                "adv_g": 0, "adv_d_fake": 0, "adv_d_real": 0
            }
            step = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="step",
                leave=True,
            )

            for batch_data in progress_bar:
                step += 1
                images = batch_data["image"].to(self.device)

                g_losses = self._train_generator_step(images, epoch)
                d_losses = self._train_discriminator_step(
                    g_losses["reconstruction_tensor"], images, epoch
                )

                epoch_losses["g_loss"] += g_losses["total"]
                epoch_losses["pixel_wise"] += g_losses["pixel-wise"]
                epoch_losses["perceptual"] += g_losses["perceptual"]
                epoch_losses["ssim"] += g_losses["ssim"]
                epoch_losses["focal_frequency"] += g_losses["focal_frequency"]
                epoch_losses["kl"] += g_losses["kl"]
                epoch_losses["adv_g"] += g_losses["adversarial"]
                epoch_losses["d_loss"] += d_losses["total"]
                epoch_losses["adv_d_fake"] += d_losses["fake"]
                epoch_losses["adv_d_real"] += d_losses["real"]

                progress_bar.set_postfix(
                    {
                        "G_loss": f"{g_losses['total']:.4f}",
                        "D_loss": f"{d_losses['total']:.4f}",
                        "Pixel-wise": f"{g_losses['pixel-wise']:.4f}",
                        "Perceptual": f"{g_losses['perceptual']:.4f}",
                        "SSIM": f"{g_losses['ssim']:.4f}",
                        "Focal Frequency": f"{g_losses['focal_frequency']:.4f}",
                        "KL": f"{g_losses['kl']:.4f}",
                    }
                )

            progress_bar.close()
            epoch_time = time.time() - epoch_start

            # Log average training losses
            avg_losses = {k: v / step for k, v in epoch_losses.items()}
            train_metrics = {
                "train_generator_loss": avg_losses["g_loss"],
                "train_discriminator_loss": avg_losses["d_loss"],
                "train_pixel_wise_loss": avg_losses["pixel_wise"],
                "train_kl_loss": avg_losses["kl"],
                "train_perceptual_loss": avg_losses["perceptual"],
                "train_ssim_loss": avg_losses["ssim"],
                "train_focal_frequency_loss": avg_losses["focal_frequency"],
                "train_adv_generator_loss": avg_losses["adv_g"],
                "train_adv_discriminator_fake_loss": avg_losses["adv_d_fake"],
                "train_adv_discriminator_real_loss": avg_losses["adv_d_real"],
                "epoch_time_seconds": epoch_time,
            }
            mlflow.log_metrics(train_metrics, step=epoch)

            tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs} Summary:")
            tqdm.write(
                f"  Avg Train G Loss: {avg_losses['g_loss']:.4f}, Avg Train D Loss: {avg_losses['d_loss']:.4f}"
            )
            tqdm.write(
                f"  Avg Pixel-wise Loss: {avg_losses['pixel_wise']:.4f}, Avg Perceptual Loss: {avg_losses['perceptual']:.4f}, Avg SSIM Loss: {avg_losses['ssim']:.4f}, Avg KL Loss: {avg_losses['kl']:.4f}"
            )
            tqdm.write(f"  Time: {epoch_time:.2f}s")

            # Validation step
            if (epoch + 1) % self.val_interval == 0:
                val_epoch_start = time.time()
                avg_val_recon_loss = self._validate_epoch(epoch)
                val_epoch_time = time.time() - val_epoch_start

                mlflow.log_metrics({
                    "learning_rate_g": self.scheduler_g.optimizer.param_groups[0]['lr'],
                    "val_epoch_time": val_epoch_time
                }, step=epoch)

                self.scheduler_g.step(avg_val_recon_loss)

                if avg_val_recon_loss < best_val_metric:
                    best_val_metric = avg_val_recon_loss
                    self._save_best_models(best_val_metric, epoch)

            # Save checkpoint every save_interval epochs
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch)

        # Test
        test_recon_loss, test_metrics = self._test_epoch()
        mlflow.log_metrics({"test_recon_loss": test_recon_loss}, step=0)
        mlflow.log_metrics(test_metrics, step=0)

        tqdm.write(f"Test Recon Loss: {test_recon_loss:.4f}")
        for key, value in test_metrics.items():
            tqdm.write(f"Test {key}: {value:.4f}")

        # Log final models
        mlflow.log_metric("final_test_recon_loss", test_recon_loss)
        mlflow.log_metrics(test_metrics, step=0)

        # Save final models
        mlflow.log_metric("final_epoch", self.num_epochs)
        mlflow.log_metric("final_val_recon_loss", best_val_metric)

        tqdm.write("Saving final models...")
        mlflow.pytorch.log_model(self.autoencoder, "final_autoencoder_model")
        mlflow.pytorch.log_model(self.discriminator, "final_discriminator_model")

        # End of Training
        final_autoencoder_path = "final_autoencoder_model.pth"
        final_discriminator_path = "final_discriminator_model.pth"

        torch.save(self.autoencoder.state_dict(), final_autoencoder_path)
        torch.save(self.discriminator.state_dict(), final_discriminator_path)

        mlflow.log_artifact(final_autoencoder_path, artifact_path="models")
        mlflow.log_artifact(final_discriminator_path, artifact_path="models")

        mlflow.pytorch.log_model(self.autoencoder, "autoencoder_model")
        mlflow.pytorch.log_model(self.discriminator, "discriminator_model")

        mlflow.end_run()
        print("Training finished.")
