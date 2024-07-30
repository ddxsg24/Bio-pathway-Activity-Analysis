props <-
function(healthy_dat, dat, pathway_edges = NULL, batch_correct = FALSE, healthy_batches = NULL, dat_batches = NULL){
  
  #check if user wants to batch correct
  if(batch_correct){
    if(is.null(dat_batches)){
      stop("Data batches not provided.")
    }else if(!is.numeric(dat_batches)){
      stop("Data batches are not numeric.")
    }else if(length(dat_batches) != nrow(dat)){
      stop("Data batches length does not match number of samples in data.")
    }else if(is.null(healthy_batches)){
      stop("Healthy data batches not provided.")
    }else if(!is.numeric(healthy_batches)){
      stop("Healthy data batches are not numeric.")
    }else if(length(healthy_batches) != nrow(healthy_dat)){
      stop("Healthy data batches length does not match number of samples in healthy data.")
    }else{
      data_combat <- ComBat(t(rbind(healthy_dat, dat)), batch = c(healthy_batches, dat_batches))
      data_combat <- t(data_combat)
      healthy_dat <- as.data.frame(data_combat[1:nrow(healthy_dat),])
      dat <- as.data.frame(data_combat[(nrow(healthy_dat)+1):nrow(data_combat),])
    }
  }
  
  #check for user input edges
  if(is.null(pathway_edges)){
    data(kegg_pathway_edges)
    pathway_edges <- kegg_pathway_edges
  }else if(!is.data.frame(pathway_edges)){
    stop("Error: Pathway edges provided are not in a data frame.")
  }
  
  #check for data structure and gene IDs
  if(!(is(healthy_dat, "ExpressionSet") | is.data.frame(healthy_dat))){
    stop("Healthy data provided is not a data frame or ExpressionSet.")
  }
  if(!(is(dat, "ExpressionSet") | is.data.frame(dat))){
    stop("Data provided is not a data frame or ExpressionSet.")
  }
  if(is(healthy_dat, "ExpressionSet")){
    healthy_dat = as.data.frame(t(exprs(healthy_dat)))
  }
  if(is(dat, "ExpressionSet")){
    dat = as.data.frame(t(exprs(dat)))
  }
  if (sum(grepl("[^0-9]", colnames(dat))) > 0 ){
    stop("Data provided does not have Entrez ID as gene identifier column names.")
  }
  if (sum(grepl("[^0-9]", colnames(healthy_dat))) > 0 ){
    stop("Healthy data provided does not have Entrez ID as gene identifier column names.")
  }
  
  #check number of columns of healthy data and data are the same
  #if not the same, give warning and take common set
  if (ncol(healthy_dat) != ncol(dat)){
    warning("Healthy data and data have a different number of genes. Using the intersection.")
  }else if (sum(colnames(healthy_dat) %in% colnames(dat)) != ncol(healthy_dat)){
    warning("Healthy data and data have different genes. Using the intersection.")
  }
  
  common <- intersect(colnames(healthy_dat), colnames(dat))
  healthy_dat <- healthy_dat[,colnames(healthy_dat) %in% common]
  dat <- dat[,colnames(dat) %in% common]
  
  LL <- data.frame(stringsAsFactors = FALSE)
  all_pathways = unique(pathway_edges[,3])
  for (i in 1:length(all_pathways)){
    edgelist <- pathway_edges[pathway_edges[,3] == all_pathways[i],]
    
    edgelist <- edgelist[(edgelist[,2] %in% colnames(healthy_dat)) & (edgelist[,1] %in% colnames(healthy_dat)),]
    
    if(nrow(edgelist) > 0){ #proceed only if pathway has at least one edge
      ordering <- sample(1:nrow(edgelist), nrow(edgelist), replace = FALSE)
      pathway_graph <- empty.graph(unique(c(edgelist[,1], edgelist[,2])))
      for (j in ordering){
        pathway_graph <- tryCatch({ #add edges in random order, exclude cycles
          set.arc(pathway_graph, from = edgelist[j, 1], to = edgelist[j, 2], debug = FALSE)
        }, error = function(err) {pathway_graph})
      }
      
      #calculate log-likelihood
      pathway_bn <- bn.fit(pathway_graph, as.data.frame(healthy_dat[,colnames(healthy_dat) %in% c(edgelist[,1], edgelist[,2])]))
      sample_LL <- logLik(pathway_bn, dat[,colnames(dat) %in% c(edgelist[,1], edgelist[,2])], by.sample = TRUE)
      if (!is.na(sum(sample_LL))){
        LL <- rbind(LL, data.frame(pathway_ID = all_pathways[i], t(sample_LL), stringsAsFactors = FALSE))
      }
    }
    
  }
  
  colnames(LL) = c("pathway_ID", rownames(dat))
  return(LL)
}
