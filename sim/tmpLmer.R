## utility f'n for checking starting values
getStart <- function(start, pred, returnVal = c("theta","all")) {
  returnVal <- match.arg(returnVal)
  doFixef <- returnVal == "all"
  ## default values
  theta <- pred$theta
  fixef <- pred$delb
  if (is.numeric(start)) {
    theta <- start
  } else if (is.list(start)) {
    if (!all(vapply(start, is.numeric, NA)))
      stop("all elements of start must be numeric")
    if (length((badComp <- setdiff(names(start), c("theta","fixef")))) > 0)
      stop("incorrect components in start list: ", badComp)
    if (!is.null(start$theta)) theta <- start$theta
    if (doFixef) {
      noBeta <- is.null(start$beta)
      if(!is.null(sFE <- start$fixef) || !noBeta) {
        fixef <-
          if(!is.null(sFE)) {
            if(!noBeta)
              message("Starting values for fixed effects coefficients",
                      "specified through both 'fixef' and 'beta',",
                      "only 'fixef' used")
            ## FIXME? accumulating heuristic evidence for drop1()-like case
            if((p <- length(fixef)) < length(sFE) && p == ncol(pred$X) &&
               is.character(ns <- names(sFE)) &&
               all((cnX <- dimnames(pred$X)[[2L]]) %in% ns))
              ## take "matching" fixef[] only
              sFE[cnX]
            else
              sFE
          }
        else if(!noBeta)
          start$beta
      }
      if (length(fixef)!=length(pred$delb))
        stop("incorrect number of fixef components (!=",length(pred$delb),")")
    }
  }
  else if (!is.null(start))
    stop("'start' must be NULL, a numeric vector or named list of such vectors")
  if (!is.null(start) && length(theta) != length(pred$theta))
    stop("incorrect number of theta components (!=",length(pred$theta),")")
  
  if(doFixef) c(theta, fixef) else theta
}

tmp <- optimizeLmer(m2)

optwrap2 <- function(optimizer, fn, par, lower = -Inf, upper = Inf,
                    control = list(), adj = FALSE, calc.derivs = TRUE,
                    use.last.params = FALSE,
                    verbose = 0L)
{
  ## control must be specified if adj==TRUE;
  ##  otherwise this is a fairly simple wrapper
  optfun <- getOptfun(optimizer)
  optName <- if(is.character(optimizer)) optimizer
  else ## "good try":
    deparse(substitute(optimizer))[[1L]]
  
  lower <- rep_len(lower, length(par))
  upper <- rep_len(upper, length(par))
  
  if (adj)
    ## control parameter tweaks: only for second round in nlmer, glmer
    switch(optName,
           "bobyqa" = {
             if(!is.numeric(control$rhobeg)) control$rhobeg <- 0.0002
             if(!is.numeric(control$rhoend)) control$rhoend <- 2e-7
           },
           "Nelder_Mead" = {
             if (is.null(control$xst))  {
               thetaStep <- 0.1
               nTheta <- length(environment(fn)$pp$theta)
               betaSD <- sqrt(diag(environment(fn)$pp$unsc()))
               control$xst <- 0.2* c(rep.int(thetaStep, nTheta),
                                     pmin(betaSD, 10))
             }
             if (is.null(control$xt)) control$xt <- control$xst*5e-4
           })
  switch(optName,
         "bobyqa" = {
           if(all(par == 0)) par[] <- 0.001  ## minor kludge
           if(!is.numeric(control$iprint)) control$iprint <- min(verbose, 3L)
         },
         "Nelder_Mead" = control$verbose <- verbose,
         "nloptwrap" = control$print_level <- min(as.numeric(verbose),3L),
         ## otherwise:
         if(verbose) warning(gettextf(
           "'verbose' not yet passed to optimizer '%s'; consider fixing optwrap()",
           optName), domain = NA)
  )
  arglist <- list(fn = fn, par = par, lower = lower, upper = upper, control = control)
  ## optimx: must pass method in control (?) because 'method' was previously
  ## used in lme4 to specify REML vs ML
  if (optName == "optimx") {
    if (is.null(method <- control$method))
      stop("must specify 'method' explicitly for optimx")
    arglist$control$method <- NULL
    arglist <- c(arglist, list(method = method))
  }
  ## FIXME: test!  effects of multiple warnings??
  ## may not need to catch warnings after all??
  curWarnings <- list()
  opt <- withCallingHandlers(do.call(optfun, arglist),
                             warning = function(w) {
                               curWarnings <<- append(curWarnings,list(w$message))
                             })
  ## cat("***",unlist(tail(curWarnings,1)))
  ## FIXME: set code to warn on convergence !=0
  ## post-fit tweaking
  if (optName == "bobyqa") {
    opt$convergence <- opt$ierr
  }
  else if (optName == "optimx") {
    opt <- list(par = coef(opt)[1,],
                fvalues = opt$value[1],
                method = method,
                conv = opt$convcode[1],
                feval = opt$fevals + opt$gevals,
                message = attr(opt,"details")[,"message"][[1]])
  }
  if ((optconv <- getConv(opt)) != 0) {
    wmsg <- paste("convergence code",optconv,"from",optName)
    if (!is.null(opt$msg)) wmsg <- paste0(wmsg,": ",opt$msg)
    warning(wmsg)
    curWarnings <<- append(curWarnings,list(wmsg))
  }
  ## pp_before <- environment(fn)$pp
  ## save(pp_before,file="pp_before.RData")
  
  if (calc.derivs) {
    if (use.last.params) {
      ## +0 tricks R into doing a deep copy ...
      ## otherwise element of ref class changes!
      ## FIXME:: clunky!!
      orig_pars <- opt$par
      orig_theta <- environment(fn)$pp$theta+0
      orig_pars[seq_along(orig_theta)] <- orig_theta
    }
    if (verbose > 10) cat("computing derivatives\n")
    derivs <- deriv12(fn,opt$par,fx = opt$value)
    if (use.last.params) {
      ## run one more evaluation of the function at the optimized
      ##  value, to reset the internal/environment variables in devfun ...
      fn(orig_pars)
    }
  } else derivs <- NULL
  
  if (!use.last.params) {
    ## run one more evaluation of the function at the optimized
    ##  value, to reset the internal/environment variables in devfun ...
    fn(opt$par)
  }
  structure(opt, ## store all auxiliary information
            optimizer = optimizer,
            control   = control,
            warnings  = curWarnings,
            derivs    = derivs)
}

optimizeLmer2 <- function(devfun, optimizer = formals(lmerControl)$optimizer, 
          restart_edge = formals(lmerControl)$restart_edge, boundary.tol = formals(lmerControl)$boundary.tol, 
          start = NULL, verbose = 0L, control = list(), ...) 
{
  verbose <- as.integer(verbose)
  rho <- environment(devfun)
  opt <- optwrap2(optimizer, devfun, getStart(start, rho$lower, 
                                             rho$pp), lower = rho$lower, control = control, adj = FALSE, 
                 verbose = verbose, ...)
  if (restart_edge) {
    if (length(bvals <- which(rho$pp$theta == rho$lower)) > 
        0) {
      theta0 <- new("numeric", rho$pp$theta)
      d0 <- devfun(theta0)
      btol <- 1e-05
      bgrad <- sapply(bvals, function(i) {
        bndval <- rho$lower[i]
        theta <- theta0
        theta[i] <- bndval + btol
        (devfun(theta) - d0)/btol
      })
      devfun(theta0)
      if (any(bgrad < 0)) {
        if (verbose) 
          message("some theta parameters on the boundary, restarting")
        opt <- optwrap2(optimizer, devfun, opt$par, lower = rho$lower, 
                       control = control, adj = FALSE, verbose = verbose, 
                       ...)
      }
    }
  }
  if (boundary.tol > 0) 
    check.boundary(rho, opt, devfun, boundary.tol)
  else opt
}

## Internal utility, only used in optwrap() :
##' @title Get the optimizer function and check it minimally
##' @param optimizer character string ( = function name) *or* function
getOptfun <- function(optimizer) {
  if (((is.character(optimizer) && optimizer == "optimx") ||
       deparse(substitute(optimizer)) == "optimx")) {
    if (!requireNamespace("optimx")) {
      stop(shQuote("optimx")," package must be installed order to ",
           "use ",shQuote('optimizer="optimx"'))
    }
    optfun <- optimx::optimx
  } else if (is.character(optimizer)) {
    optfun <- tryCatch(get(optimizer), error = function(e) NULL)
  } else optfun <- optimizer
  if (is.null(optfun)) stop("couldn't find optimizer function ",optimizer)
  if (!is.function(optfun)) stop("non-function specified as optimizer")
  needArgs <- c("fn","par","lower","control")
  if (anyNA(match(needArgs, names(formals(optfun)))))
    stop("optimizer function must use (at least) formal parameters ",
         paste(sQuote(needArgs), collapse = ", "))
  optfun
}
