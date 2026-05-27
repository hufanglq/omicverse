from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
from anndata import AnnData


def test_space_exports_gsmap_symbol():
    import omicverse as ov

    assert "gsmap" in ov.space.__all__
    assert hasattr(ov.space, "gsmap")


def test_gsmap_runner_builds_find_latent_config(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation=None)

    config = runner.build_find_latent_config(data_layer="X", n_comps=8)
    assert config.sample_name == "toy"
    assert config.data_layer == "X"
    assert config.n_comps == 8


def test_gsmap_runner_writes_input_h5ad_before_pipeline(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy")

    path = runner.prepare_input_h5ad()
    assert path.exists()
    assert path.suffix == ".h5ad"


def test_vendored_gsmap_core_imports_without_cli_layer():
    from omicverse.external.gsmap import (
        FindLatentRepresentationsConfig,
        LatentToGeneConfig,
        run_find_latent_representation,
        run_latent_to_gene,
    )

    assert FindLatentRepresentationsConfig is not None
    assert LatentToGeneConfig is not None
    assert callable(run_find_latent_representation)
    assert callable(run_latent_to_gene)


def test_pyproject_declares_gsmap_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    assert "gsmap" in optional_dependencies

    gsmap_dependencies = optional_dependencies["gsmap"]
    assert any(dependency.startswith("torch>=") for dependency in gsmap_dependencies)
    assert any(dependency.startswith("torch-geometric>=") for dependency in gsmap_dependencies)
    assert any(dependency.startswith("pyranges>=") for dependency in gsmap_dependencies)
    assert any(dependency.startswith("pyarrow>=") for dependency in gsmap_dependencies)


def test_gsmap_is_registered_in_registry():
    import omicverse as ov

    _ = ov.space.gsmap
    result = ov.find_function("gsmap")
    assert result is not None


def test_gsmap_runner_calls_external_find_latent_representation(tmp_path, monkeypatch):
    import omicverse.external.gsmap as external_gsmap
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy")
    called = {}

    def fake_run(config):
        called["config"] = config
        return "ok"

    monkeypatch.setattr(external_gsmap, "run_find_latent_representation", fake_run)

    result = runner.find_latent_representation(data_layer="X", n_comps=12)

    assert result == called["config"].hdf5_with_latent_path
    assert called["config"].data_layer == "X"
    assert called["config"].n_comps == 12


def test_gsmap_runner_builds_latent_to_gene_config(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation=None)

    config = runner.build_latent_to_gene_config(
        latent_representation="latent_GVAE",
        num_neighbour=3,
        num_neighbour_spatial=4,
    )
    assert config.sample_name == "toy"
    assert config.latent_representation == "latent_GVAE"
    assert config.num_neighbour == 3


def test_gsmap_runner_calls_external_latent_to_gene(tmp_path, monkeypatch):
    import omicverse.external.gsmap as external_gsmap
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy")
    called = {}

    def fake_run(config):
        called["config"] = config
        return "ok"

    monkeypatch.setattr(external_gsmap, "run_latent_to_gene", fake_run)

    result = runner.latent_to_gene(latent_representation="latent_GVAE", num_neighbour=3)

    assert result == called["config"].mkscore_feather_path
    assert called["config"].latent_representation == "latent_GVAE"
    assert called["config"].num_neighbour == 3


def test_gsmap_runner_builds_generate_ldscore_config(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation=None)

    resource_dir = tmp_path / "resource"
    (resource_dir / "quick_mode" / "baseline").mkdir(parents=True)
    (resource_dir / "quick_mode" / "SNP_gene_pair").mkdir(parents=True)

    config = runner.build_generate_ldscore_config(gsmap_resource_dir=str(resource_dir))
    assert config.sample_name == "toy"
    assert config.ldscore_save_format == "quick_mode"
    assert config.baseline_annotation_dir == resource_dir / "quick_mode" / "baseline"
    assert config.snp_gene_pair_dir == resource_dir / "quick_mode" / "SNP_gene_pair"


def test_vendored_generate_ldscore_quick_mode_links_resources(tmp_path):
    from omicverse.external.gsmap import GenerateLdscoreConfig, run_generate_ldscore

    baseline_dir = tmp_path / "resource" / "quick_mode" / "baseline"
    snp_gene_pair_dir = tmp_path / "resource" / "quick_mode" / "SNP_gene_pair"
    baseline_dir.mkdir(parents=True)
    snp_gene_pair_dir.mkdir(parents=True)

    config = GenerateLdscoreConfig(
        workdir=str(tmp_path),
        sample_name="toy",
        ldscore_save_format="quick_mode",
        baseline_annotation_dir=str(baseline_dir),
        snp_gene_pair_dir=str(snp_gene_pair_dir),
    )

    run_generate_ldscore(config)

    generated_dir = tmp_path / "toy" / "generate_ldscore"
    assert generated_dir.exists()
    assert (generated_dir / "baseline").exists()
    assert (generated_dir / "SNP_gene_pair").exists()
    assert (generated_dir / "toy_generate_ldscore.done").exists()


def test_gsmap_runner_builds_spatial_ldsc_config(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation=None)

    resource_dir = tmp_path / "resource"
    (resource_dir / "quick_mode").mkdir(parents=True)
    (resource_dir / "LDSC_resource" / "weights_hm3_no_hla").mkdir(parents=True)
    (resource_dir / "quick_mode" / "snp_gene_weight_matrix.h5ad").touch()
    sumstats_file = tmp_path / "toy.sumstats.gz"
    sumstats_file.write_text("SNP Z N\nrs1 0.1 1000\n", encoding="utf-8")

    config = runner.build_spatial_ldsc_config(
        gsmap_resource_dir=str(resource_dir),
        sumstats_file=str(sumstats_file),
        trait_name="trait_x",
    )

    assert config.sample_name == "toy"
    assert config.trait_name == "trait_x"
    assert config.ldscore_save_format == "quick_mode"
    assert config.snp_gene_weight_adata_path == resource_dir / "quick_mode" / "snp_gene_weight_matrix.h5ad"
    assert str(config.w_file).endswith("weights.")


def test_gsmap_runner_calls_external_spatial_ldsc(tmp_path, monkeypatch):
    import omicverse.external.gsmap as external_gsmap
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy")
    called = {}

    resource_dir = tmp_path / "resource"
    (resource_dir / "quick_mode").mkdir(parents=True)
    (resource_dir / "LDSC_resource" / "weights_hm3_no_hla").mkdir(parents=True)
    (resource_dir / "quick_mode" / "snp_gene_weight_matrix.h5ad").touch()
    sumstats_file = tmp_path / "toy.sumstats.gz"
    sumstats_file.write_text("SNP Z N\nrs1 0.1 1000\n", encoding="utf-8")

    def fake_run(config):
        called["config"] = config
        return "ok"

    monkeypatch.setattr(external_gsmap, "run_spatial_ldsc", fake_run)

    result = runner.spatial_ldsc(
        gsmap_resource_dir=str(resource_dir),
        sumstats_file=str(sumstats_file),
        trait_name="trait_x",
    )

    assert result == called["config"].ldsc_save_dir
    assert called["config"].trait_name == "trait_x"
    assert called["config"].ldscore_save_format == "quick_mode"


def test_gsmap_runner_builds_cauchy_combination_config(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obs["celltype"] = ["a", "a", "b", "b"]
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation="celltype")

    config = runner.build_cauchy_combination_config(trait_name="trait_x")

    assert config.sample_name == "toy"
    assert config.trait_name == "trait_x"
    assert config.annotation == "celltype"
    assert str(config.output_file).endswith("toy_trait_x.Cauchy.csv.gz")


def test_gsmap_runner_calls_external_cauchy_combination(tmp_path, monkeypatch):
    import omicverse.external.gsmap as external_gsmap
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obs["celltype"] = ["a", "a", "b", "b"]
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation="celltype")
    called = {}

    def fake_run(config):
        called["config"] = config
        return "ok"

    monkeypatch.setattr(external_gsmap, "run_cauchy_combination", fake_run)

    result = runner.cauchy_combination(trait_name="trait_x")

    assert result == called["config"].output_file
    assert called["config"].trait_name == "trait_x"
    assert called["config"].annotation == "celltype"


def test_vendored_cauchy_combination_writes_result_file(tmp_path):
    import pandas as pd
    import scanpy as sc
    from omicverse.external.gsmap import CauchyCombinationConfig, run_cauchy_combination

    adata = AnnData(np.ones((4, 3)))
    adata.obs["celltype"] = ["a", "a", "b", "b"]
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)

    latent_dir = tmp_path / "toy" / "find_latent_representations"
    latent_dir.mkdir(parents=True)
    adata.write_h5ad(latent_dir / "toy_add_latent.h5ad")

    ldsc_dir = tmp_path / "toy" / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)
    ldsc_df = pd.DataFrame(
        {
            "spot": ["0", "1", "2", "3"],
            "beta": [0.1, 0.2, 0.3, 0.4],
            "se": [0.01, 0.02, 0.03, 0.04],
            "z": [1.0, 2.0, 3.0, 4.0],
            "p": [0.1, 0.2, 0.01, 0.02],
        }
    )
    ldsc_df.to_csv(ldsc_dir / "toy_trait_x.csv.gz", index=False, compression="gzip")

    updated = sc.read_h5ad(latent_dir / "toy_add_latent.h5ad")
    updated.obs_names = ["0", "1", "2", "3"]
    updated.write_h5ad(latent_dir / "toy_add_latent.h5ad")

    config = CauchyCombinationConfig(
        workdir=str(tmp_path),
        sample_name="toy",
        trait_name="trait_x",
        annotation="celltype",
    )

    result = run_cauchy_combination(config)

    assert Path(config.output_file).exists()
    assert list(result.columns) == ["annotation", "p_cauchy", "p_median"]
    assert set(result["annotation"]) == {"a", "b"}


def test_gsmap_runner_builds_report_config(tmp_path):
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obs["celltype"] = ["a", "a", "b", "b"]
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation="celltype")

    sumstats_file = tmp_path / "toy.sumstats.gz"
    sumstats_file.write_text("SNP\tZ\tN\nrs1\t0.1\t1000\n", encoding="utf-8")

    config = runner.build_report_config(
        trait_name="trait_x",
        sumstats_file=str(sumstats_file),
    )

    assert config.sample_name == "toy"
    assert config.trait_name == "trait_x"
    assert config.annotation == "celltype"
    assert config.sumstats_file == str(sumstats_file)
    assert str(config.get_gsmap_report_file("trait_x")).endswith(
        "toy_trait_x_gsMap_Report.html"
    )


def test_gsmap_runner_calls_external_report(tmp_path, monkeypatch):
    import omicverse.external.gsmap as external_gsmap
    from omicverse.space._gsmap import gsmap

    adata = AnnData(np.ones((4, 3)))
    adata.obs["celltype"] = ["a", "a", "b", "b"]
    adata.obsm["spatial"] = np.arange(8).reshape(4, 2)
    runner = gsmap(adata, workdir=str(tmp_path), sample_name="toy", annotation="celltype")
    called = {}

    sumstats_file = tmp_path / "toy.sumstats.gz"
    sumstats_file.write_text("SNP\tZ\tN\nrs1\t0.1\t1000\n", encoding="utf-8")

    def fake_run(config):
        called["config"] = config
        return "ok"

    monkeypatch.setattr(external_gsmap, "run_report", fake_run)

    result = runner.report(
        trait_name="trait_x",
        sumstats_file=str(sumstats_file),
    )

    assert result == called["config"].get_gsmap_report_file("trait_x")
    assert called["config"].trait_name == "trait_x"
    assert called["config"].annotation == "celltype"


def test_vendored_report_writes_html_file(tmp_path, monkeypatch):
    import pandas as pd
    from omicverse.external.gsmap import ReportConfig, run_report

    sample_dir = tmp_path / "toy"
    report_dir = sample_dir / "report" / "trait_x"
    manhattan_dir = report_dir / "manhattan_plot"
    gsmap_plot_dir = report_dir / "gsMap_plot"
    gss_plot_dir = report_dir / "GSS_plot"
    cauchy_dir = sample_dir / "cauchy_combination"

    manhattan_dir.mkdir(parents=True)
    gsmap_plot_dir.mkdir(parents=True)
    gss_plot_dir.mkdir(parents=True)
    cauchy_dir.mkdir(parents=True)

    sumstats_file = tmp_path / "toy.sumstats.gz"
    sumstats_file.write_text("SNP\tZ\tN\nrs1\t0.1\t1000\n", encoding="utf-8")

    cauchy_df = pd.DataFrame(
        {
            "annotation": ["a", "b"],
            "p_cauchy": [0.01, 0.02],
            "p_median": [0.05, 0.06],
        }
    )
    cauchy_df.to_csv(
        cauchy_dir / "toy_trait_x.Cauchy.csv.gz",
        index=False,
        compression="gzip",
    )

    gene_info_df = pd.DataFrame(
        {
            "Gene": ["gene_a", "gene_b"],
            "Annotation": ["a", "b"],
            "Median_GSS": [0.4, 0.5],
            "PCC": [0.7, 0.6],
        }
    )
    gene_info_df.to_csv(report_dir / "toy_trait_x_Gene_Diagnostic_Info.csv", index=False)

    (report_dir / "GSS_plot" / "plot_genes.csv").write_text("gene_a\n", encoding="utf-8")
    (gss_plot_dir / "toy_gene_a_Expression_Distribution.png").write_bytes(b"png")
    (gss_plot_dir / "toy_gene_a_GSS_Distribution.png").write_bytes(b"png")
    (gsmap_plot_dir / "toy_trait_x_gsMap_plot.html").write_text(
        "<div>gsmap plot</div>",
        encoding="utf-8",
    )
    (manhattan_dir / "toy_trait_x_Diagnostic_Manhattan_Plot.html").write_text(
        "<div>manhattan plot</div>",
        encoding="utf-8",
    )

    def fake_run_diagnosis(config):
        return None

    monkeypatch.setattr("omicverse.external.gsmap.report.run_diagnosis", fake_run_diagnosis)

    config = ReportConfig(
        workdir=str(tmp_path),
        sample_name="toy",
        annotation="celltype",
        trait_name="trait_x",
        sumstats_file=str(sumstats_file),
    )

    report_file = run_report(config)

    assert report_file.exists()
    report_html = report_file.read_text(encoding="utf-8")
    assert "Genetic Spatial Mapping Report" in report_html
    assert "gene_a" in report_html
    assert "manhattan plot" in report_html


def test_vendored_find_latent_representation_writes_latent_h5ad(tmp_path):
    import scanpy as sc
    from omicverse.external.gsmap import (
        FindLatentRepresentationsConfig,
        run_find_latent_representation,
    )

    adata = AnnData(
        np.random.poisson(lam=3, size=(12, 16)).astype(np.float32),
    )
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32)]
    )

    input_h5ad_path = tmp_path / "toy_input.h5ad"
    adata.write_h5ad(input_h5ad_path)

    config = FindLatentRepresentationsConfig(
        workdir=str(tmp_path),
        sample_name="toy",
        input_hdf5_path=str(input_h5ad_path),
        data_layer="X",
        epochs=1,
        feat_cell=8,
        feat_hidden1=16,
        feat_hidden2=8,
        gat_hidden1=4,
        gat_hidden2=3,
        n_comps=4,
        n_neighbors=4,
        nheads=1,
    )

    run_find_latent_representation(config)

    assert config.hdf5_with_latent_path.exists()

    output_adata = sc.read_h5ad(config.hdf5_with_latent_path)
    assert "latent_GVAE" in output_adata.obsm
    assert "latent_PCA" in output_adata.obsm
    assert output_adata.obsm["latent_GVAE"].shape[0] == output_adata.n_obs


def test_vendored_latent_to_gene_writes_marker_score_feather(tmp_path):
    import pandas as pd
    from omicverse.external.gsmap import (
        FindLatentRepresentationsConfig,
        LatentToGeneConfig,
        run_find_latent_representation,
        run_latent_to_gene,
    )

    adata = AnnData(
        np.random.poisson(lam=3, size=(12, 16)).astype(np.int32),
    )
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32)]
    )

    input_h5ad_path = tmp_path / "toy_input_latent_to_gene.h5ad"
    adata.write_h5ad(input_h5ad_path)

    latent_config = FindLatentRepresentationsConfig(
        workdir=str(tmp_path),
        sample_name="toy",
        input_hdf5_path=str(input_h5ad_path),
        data_layer="X",
        epochs=1,
        feat_cell=8,
        feat_hidden1=16,
        feat_hidden2=8,
        gat_hidden1=4,
        gat_hidden2=3,
        n_comps=4,
        n_neighbors=4,
        nheads=1,
    )
    run_find_latent_representation(latent_config)

    latent_to_gene_config = LatentToGeneConfig(
        workdir=str(tmp_path),
        sample_name="toy",
        input_hdf5_path=str(latent_config.hdf5_with_latent_path),
        latent_representation="latent_GVAE",
        num_neighbour=3,
        num_neighbour_spatial=4,
    )

    run_latent_to_gene(latent_to_gene_config)

    assert latent_to_gene_config.mkscore_feather_path.exists()

    mk_score_df = pd.read_feather(latent_to_gene_config.mkscore_feather_path)
    assert "HUMAN_GENE_SYM" in mk_score_df.columns
    assert mk_score_df.shape[1] > 1
