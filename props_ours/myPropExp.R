rm(list=ls())
cat("\014")
# source("http://bioconductor.org/biocLite.R") 
# biocLite()

# source("props.R")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("PROPS")

library(PROPS)

library(readxl)
setwd('C:/Users/50209/Desktop/Pathway-based disease classification/code/props_my')
gene_expression_normal <- data.frame(read_excel("./data/expression data/Allgene7_normal.xlsx"))
# #gene_expression_disease <- data.frame(read_excel("./data/expression data/Allgene7_GSE9686.xlsx"))
# #gene_expression_disease <- data.frame(read_excel("./data/expression data/Allgene7_GSE10616.xlsx"))
# #gene_expression_disease <- data.frame(read_excel("./data/expression data/Allgene7_GSE16879.xlsx"))
# #gene_expression_disease <- data.frame(read_excel("./data/expression data/Allgene7_GSE36807.xlsx"))
gene_expression_disease <- data.frame(read_excel("./data/expression data/Allgene7_GSE71730.xlsx"))

setwd('D:/XY_work/project/PROPS/props_my')
pathway.dir <-"./data/pathway data"

pathway.files <- dir(pathway.dir, full.names = T)
library(readr)

pathway.all =list()
k=1
for (file in pathway.files) {
  print(file)
  hsa <- data.frame(read_table2(file,col_names = FALSE))
  hsa$X1 <- as.character(hsa$X1)
  hsa$X2 <- as.character(hsa$X2)
  hsa$'X3' <- as.character(k)
  k=k+1
  
  pathway.all <-rbind(pathway.all,hsa)
}
kegg_pathway_edges <- pathway.all

library(data.table)
# transpose
healthy_dat <- transpose(gene_expression_normal)
# get row and colnames in order
colnames(healthy_dat) <- rownames(gene_expression_normal)
rownames(healthy_dat) <- colnames(gene_expression_normal)

# transpose
dat <- transpose(gene_expression_disease)
# get row and colnames in order
colnames(dat) <- rownames(gene_expression_disease)
rownames(dat) <- colnames(gene_expression_disease)

library(bnlearn)
props_result <- props(healthy_dat, dat, pathway_edges = kegg_pathway_edges)

write.table(props_result,file="C:/Users/a/Desktop/e.txt",quote = FALSE,row.names = FALSE, col.names = FALSE)


