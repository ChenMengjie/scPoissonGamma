## download .fa for each chromosome and calculate the GC content for a given list of intervals

GCcount <- function(chr, intervals, url){
  options(scipen=50)
  filename <- paste(url, chr, ".fa.gz", sep="")
  f <- tempfile()
  download.file(filename, f, method="wget")
  data <- readLines(gzfile(f))
  closeAllConnections()
  unlink(f)
  GC <- NULL
  for(i in 1:nrow(intervals)){
    start <- intervals[i, 1]
    end <- intervals[i, 2]
    line_start <- floor(start/50)
    line_end <- floor(end/50)
    num_start <- start%%50
    num_end <- end%%50

    part1 <- substr(data[line_start], num_start, 50)
    part2 <- substr(data[line_end], 1, num_end)
    sequence <- c(unlist(strsplit(toupper(part1), "")), unlist(strsplit(toupper(part2), "")))

    if(line_end - line_start >= 2){
      part3 <- data[(line_start+1):(line_end-1)]
      sequence <- c(sequence, unlist(strsplit(toupper(part3), "")))
    }

    if(length(sequence[sequence%in%c("G", "C")])!=0)
      res <- length(sequence[sequence%in%c("G", "C")])/length(sequence)
    else
      res <- 0
    GC <- c(GC, res)
  }

  final <- data.frame(chr, intervals, GC, stringsAsFactors = FALSE)
  return(final)
}



