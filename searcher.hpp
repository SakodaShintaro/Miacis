#ifndef SEARCHER_HPP
#define SEARCHER_HPP

#include"alphabeta_searcher.hpp"
#include"MCTSearcher.hpp"

#ifdef USE_MCTS
using Searcher = MCTSearcher;
#else
using Searcher = AlphaBetaSearcher;
#endif

#endif