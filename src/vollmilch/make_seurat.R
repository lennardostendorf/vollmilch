# Builds a Seurat object from the files written by `vollmilch to_seurat` (this folder).
#   Rscript make_seurat.R [folder]   ->  folder/seurat.rds
# counts = raw per-cell intensities, data = asinh(intensity / cofactor) as used by vollmilch;
# metadata = sample, position, morphology, Leiden clusters, gated populations;
# reductions = UMAPs (NA for cells outside a subsampled embedding) and "spatial": x and -y in um,
# so DimPlot(obj, reduction = "spatial") shows the tissue as in the image (metadata y_um is unflipped).
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "."
f <- function(x) file.path(dir, x)

counts <- Read10X(dir, gene.column = 1)
asinh_data <- as(readMM(gzfile(f("asinh.mtx.gz"))), "CsparseMatrix")
dimnames(asinh_data) <- dimnames(counts)

meta <- read.csv(f("metadata.csv.gz"), row.names = 1, check.names = FALSE)
for (col in grep("^(leiden_|population)", names(meta), value = TRUE)) meta[[col]] <- factor(meta[[col]])

obj <- CreateSeuratObject(counts = counts, assay = "__ASSAY__", meta.data = meta[colnames(counts), , drop = FALSE])
obj <- SetAssayData(obj, layer = "data", new.data = asinh_data)
obj$orig.ident <- factor(obj$sample)  # Seurat would cut the cell names at "_" instead

for (file in list.files(dir, pattern = "^reduction_.*\\.csv\\.gz$")) {
  name <- sub("^reduction_(.*)\\.csv\\.gz$", "\\1", file)
  emb <- as.matrix(read.csv(f(file), row.names = 1))[colnames(obj), , drop = FALSE]
  key <- paste0(gsub("[^A-Za-z0-9]", "", name), "_")
  colnames(emb) <- paste0(key, seq_len(ncol(emb)))
  obj[[name]] <- CreateDimReducObject(embeddings = emb, key = key, assay = "__ASSAY__")
}

if ("population" %in% names(obj[[]])) {
  Idents(obj) <- "population"
} else {
  leiden <- grep("^leiden_", names(obj[[]]), value = TRUE)
  Idents(obj) <- if (length(leiden)) tail(leiden, 1) else "orig.ident"
}

saveRDS(obj, f("seurat.rds"))
cat(sprintf("seurat.rds: %d cells x %d markers; reductions: %s; identities: %s\n",
            ncol(obj), nrow(obj), paste(Reductions(obj), collapse = ", "),
            paste(head(levels(Idents(obj)), 8), collapse = ", ")))
