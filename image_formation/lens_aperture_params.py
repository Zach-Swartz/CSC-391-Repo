
import numpy as np
import matplotlib.pyplot as plt


def thin_lens_zi(f, z0):
    """Compute lens-to-image distance zi from focal length f and object distance z0."""
    return f * z0 / (z0 - f)


def plot_lens_to_image_distance():
    focal_lengths = [3, 9, 50, 200]
    z0_min_factor = 1.1
    def z0_min(f): return z0_min_factor * f
    z0_max = 1e4
    plt.figure(figsize=(8, 6))

    for f in focal_lengths:
        z0_start = z0_min(f)
        if z0_start >= z0_max:
            print(
                f"Skipping f={f} mm: z0_start ({z0_start}) >= z0_max ({z0_max})")
            continue
        num_points = int((z0_max - z0_start) * 4)
        if num_points <= 0:
            print(f"Skipping f={f} mm: num_points ({num_points}) <= 0")
            continue
        z0 = np.linspace(z0_start, z0_max, num_points)
        zi = thin_lens_zi(f, z0)
        plt.loglog(z0, zi, label=f'f = {f} mm')
        plt.axvline(f, linestyle='--', label=f'z0 = f ({f} mm)')

    plt.xlabel('Object Distance $z_0$ (mm)')
    plt.ylabel('Lens-to-Image Distance $z_i$ (mm)')
    plt.ylim([0, 3000])
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Lens-to-Image Distance vs Object Distance')
    plt.show()


def plot_aperture_diameter():
    lenses = [
        {"f": 24, "N": 1.4},
        {"f": 50, "N": 1.8},
        {"f": 70, "N": 2.8},
        {"f": 200, "N": 2.8},
        {"f": 400, "N": 2.8},
        {"f": 600, "N": 4.0},
    ]
    f_values = [lens["f"] for lens in lenses]
    N_values = [lens["N"] for lens in lenses]
    D_values = [f / N for f, N in zip(f_values, N_values)]

    plt.figure(figsize=(8, 6))
    plt.plot(f_values, D_values, 'o-', label='Aperture Diameter $D$')
    for f, N, D in zip(f_values, N_values, D_values):
        plt.text(f, D, f'f/{N}', ha='right', va='bottom')

    plt.xlabel('Focal Length $f$ (mm)')
    plt.ylabel('Aperture Diameter $D$ (mm)')
    plt.title('Aperture Diameter vs Focal Length for Popular Lenses')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    print("Aperture diameters for each lens (mm):")
    for lens, D in zip(lenses, D_values):
        print(
            f"Focal length: {lens['f']} mm, f/{lens['N']}, Needed Aperture diameter: {D:.2f} mm")


if __name__ == "__main__":
    plot_lens_to_image_distance()
    plot_aperture_diameter()
