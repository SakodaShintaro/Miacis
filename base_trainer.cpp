#include"base_trainer.hpp"
#include"network.hpp"

//#define PRINT_DEBUG

#ifdef USE_NN
std::array<double, 2> BaseTrainer::addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher) {
    assert(teacher.size() == OUTPUT_DIM);

    const auto input = pos.makeFeature();
    const auto& params = pos.evalParams();
    const Vec input_vec = Eigen::Map<const Vec>(input.data(), input.size());

    Vec x[LAYER_NUM], u[LAYER_NUM];
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        x[i] = (i == 0 ? input_vec : Network::activationFunction(u[i - 1]));
        u[i] = params.w[i] * x[i] + params.b[i];
    }

    //Policy
    auto y = softmax(pos.policy());

    //Policyの損失
    double policy_loss = 0.0;
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        policy_loss += crossEntropy(y[i], teacher[i]);
    }

    //Valueの損失
#ifdef USE_CATEGORICAL
    auto v = pos.valueDist();

    double value_loss = 0.0;
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        value_loss += crossEntropy(v[i], teacher[POLICY_DIM + i]);
#ifdef PRINT_DEBUG
        std::cout << v[i] << " " << teacher[POLICY_DIM + i] << std::endl;
#endif
    }
#else
    const auto v = pos.valueForTurn();
    double value_loss = binaryCrossEntropy(v, teacher[POLICY_DIM]);
#endif

    //Policyの勾配
    Vec delta(OUTPUT_DIM);
    for (int32_t i = 0; i < POLICY_DIM; i++) {
        delta(i) = (CalcType)(POLICY_LOSS_COEFF * (y[i] - teacher[i]));
    }
    //Valueの勾配
#ifdef USE_CATEGORICAL
    for (int32_t i = 0; i < BIN_SIZE; i++) {
        delta(POLICY_DIM + i) = (CalcType)(VALUE_LOSS_COEFF * (v[i] - teacher[POLICY_DIM + i]));
    }
#else
    delta(POLICY_DIM) = (CalcType)(VALUE_LOSS_COEFF * (v - teacher[POLICY_DIM]));
#endif

#ifdef PRINT_DEBUG
    std::cout << "value   = " << delta(POLICY_DIM) << "\n" << std::endl;

    pos.print();
    for (auto move : pos.generateAllMoves()) {
        auto index = move.toLabel();
        std::cout << y[index] << " " << teacher[index] << " ";
        move.print();
    }
#endif

    //逆伝播
    for (int32_t i = LAYER_NUM - 1; i >= 0; i--) {
        grad.w[i] += delta * x[i].transpose();
        grad.b[i] += delta;
        if (i == 0) {
            break;
        }
        delta = Network::d_activationFunction(u[i - 1]).array() * (params.w[i].transpose() * delta).array();
    }

    return { policy_loss, value_loss };
}
#else
double BaseTrainer::addGrad(EvalParams<LearnEvalType>& grad, Position& pos, TeacherType teacher) {
    //評価値を取得
    auto score = (int32_t)pos.scoreForTurn();

    //損失を計算
    double win_y = sigmoid(score, CP_GAIN);
    double loss = binaryCrossEntropy(win_y, teacher);

    //勾配の変化量を計算
    auto grad_delta = (LearnEvalType)(win_y - teacher);
    //double grad_delta2 = grad_delta * d_sigmoid(score, CP_GAIN) / CP_GAIN * 4;

#ifdef PRINT_DEBUG
    pos.print();
    std::cout << "y = " << win_y << ", t = " << teacher << std::endl;
    std::cout << "loss = " << loss << ", grad_delta = " << grad_delta << std::endl;
#endif

    //後手の時は勾配の符号を反転させる
    updateGradient(grad, pos.features(), (pos.color() == BLACK ? grad_delta : -grad_delta));

    return loss;
}
#endif

#ifdef USE_NN
void BaseTrainer::verifyAddGrad(Position & pos, TeacherType teacher) {
    auto grad_bp = std::make_unique<EvalParams<LearnEvalType>>();
    auto loss = addGrad(*grad_bp, pos, teacher);

    constexpr CalcType eps = 0.001f;
    std::cout << std::fixed << std::setprecision(15);

    //値を変えずに順伝播したときの損失
    double loss1 = loss[0] + loss[1];
    auto y1 = softmax(pos.policy());
#ifdef USE_CATEGORICAL
    auto value_dist1 = pos.valueDist();
#else
    auto value1 = pos.valueForTurn();
#endif

    for (int32_t i = 0; i < LAYER_NUM; i++) {
        for (int32_t j = 0; j < MATRIX_SIZE[i][0]; j++) {
            for (int32_t k = 0; k < MATRIX_SIZE[i][1]; k++) {
                eval_params->w[i](j, k) += eps;
                pos.resetCalc();
                double loss2 = 0.0;
                auto y2 = softmax(pos.policy());
                for (int32_t l = 0; l < POLICY_DIM; l++) {
                    loss2 += crossEntropy(y2[l], teacher[l]);
                }

#ifdef USE_CATEGORICAL
                auto value_dist2 = pos.valueDist();
                for (int32_t l = 0; l < BIN_SIZE; l++) {
                    loss2 += crossEntropy(value_dist2[l], teacher[POLICY_DIM + l]);
                }
#else
                auto value2 = pos.valueForTurn();
                loss2 += binaryCrossEntropy(value2, teacher[POLICY_DIM]);
#endif
                eval_params->w[i](j, k) -= eps;

                double grad = (loss2 - loss1) / eps;

                if (abs(grad - grad_bp->w[i](j, k)) >= 0.005) {
                    printf("勾配がおかしい\n");
                    std::cout << "(i, j, k) = (" << i << ", " << j << ", " << k << ")" << std::endl;
                    std::cout << "loss1   = " << loss1 << std::endl;
                    std::cout << "loss2   = " << loss2 << std::endl;
                    std::cout << "grad    = " << grad << std::endl;
                    std::cout << "grad_bp = " << grad_bp->w[i](j, k) << std::endl;
                }
            }

            eval_params->b[i](j) += eps;
            pos.resetCalc();
            double loss2 = 0.0;
            auto y2 = softmax(pos.policy());
            for (int32_t l = 0; l < POLICY_DIM; l++) {
                loss2 += crossEntropy(y2[l], teacher[l]);
            }

#ifdef USE_CATEGORICAL
            auto value_dist2 = pos.valueDist();
            for (int32_t l = 0; l < BIN_SIZE; l++) {
                loss2 += crossEntropy(value_dist2[l], teacher[POLICY_DIM + l]);
            }
#else
            auto value2 = pos.valueForTurn();
            loss2 += binaryCrossEntropy(value2, teacher[POLICY_DIM]);
#endif
            eval_params->b[i](j) -= eps;

            double grad = (loss2 - loss1) / eps;

            if (std::abs(grad - grad_bp->b[i](j)) >= 0.005) {
                printf("勾配がおかしい\n");
                std::cout << "(i, j) = (" << i << ", " << j << ")" << std::endl;
                std::cout << "loss1   = " << loss1 << std::endl;
                std::cout << "loss2   = " << loss2 << std::endl;
                std::cout << "grad    = " << grad << std::endl;
                std::cout << "grad_bp = " << grad_bp->b[i](j) << std::endl;
            }
        }
    }
}
#endif

#ifndef USE_NN
void BaseTrainer::updateGradient(EvalParams<LearnEvalType>& grad, const Features& features, LearnEvalType delta) {
    //featuresに出てくる特徴量に関わる勾配すべてをdeltaだけ変える
    int32_t c = (features.color == BLACK ? 1 : -1);

    const int32_t bk_sq  = SquareToNum[features.king_sq[BLACK]];
    const int32_t bk_sqr = SquareToNum[InvSquare[features.king_sq[BLACK]]];
    const int32_t wk_sq  = SquareToNum[features.king_sq[WHITE]];
    const int32_t wk_sqr = SquareToNum[InvSquare[features.king_sq[WHITE]]];

    std::array<LearnEvalType, 2> d1 = {  delta, c * delta };
    std::array<LearnEvalType, 2> d2 = { -delta, c * delta };

    for (uint32_t i = 0; i < PIECE_STATE_LIST_SIZE; i++) {
        const PieceState nor_i = features.piece_state_list[0][i];
        const PieceState inv_i = features.piece_state_list[1][i];

        //普通
        grad.kkp[bk_sq][wk_sq][nor_i] += d1;

        //180度反転
        grad.kkp[wk_sqr][bk_sqr][inv_i] += d2;

        //左右反転
        grad.kkp[mirrorSqNum(bk_sq)][mirrorSqNum(wk_sq)][mirrorPieceState(nor_i)] += d1;
        
        //180度かつ左右反転
        grad.kkp[mirrorSqNum(wk_sqr)][mirrorSqNum(bk_sqr)][mirrorPieceState(inv_i)] += d2;
        
        for (uint32_t j = i + 1; j < PIECE_STATE_LIST_SIZE; j++) {
            const PieceState nor_j = features.piece_state_list[0][j];
            const PieceState inv_j = features.piece_state_list[1][j];

            //普通
            grad.kpp[bk_sq ][nor_i][nor_j] += d1;
            grad.kpp[wk_sqr][inv_i][inv_j] += d2;

            //順番逆
            grad.kpp[bk_sq ][nor_j][nor_i] += d1;
            grad.kpp[wk_sqr][inv_j][inv_i] += d2;

            //左右反転
            grad.kpp[mirrorSqNum(bk_sq) ][mirrorPieceState(nor_i)][mirrorPieceState(nor_j)] += d1;
            grad.kpp[mirrorSqNum(wk_sqr)][mirrorPieceState(inv_i)][mirrorPieceState(inv_j)] += d2;
            
            //左右反転の順番逆
            grad.kpp[mirrorSqNum(bk_sq) ][mirrorPieceState(nor_j)][mirrorPieceState(nor_i)] += d1;
            grad.kpp[mirrorSqNum(wk_sqr)][mirrorPieceState(inv_j)][mirrorPieceState(inv_i)] += d2;
        }
    }
}
#endif

void BaseTrainer::updateParams(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad) {
    if (OPTIMIZER_NAME == "SGD") {
        updateParamsSGD(params, grad);
    } else if (OPTIMIZER_NAME == "MOMENTUM") {
        updateParamsMomentum(params, grad, *pre_update_);
    } else {
        std::cerr << "Illigal Optimizer Name : " << OPTIMIZER_NAME << std::endl;
        assert(false);
    }
}

void BaseTrainer::updateParamsSGD(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        params.w[i].array() -= LEARN_RATE * grad.w[i].array();
        params.b[i].array() -= LEARN_RATE * grad.b[i].array();
    }
#else
    for (int k1 = 0; k1 < SqNum; k1++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int p2 = 0; p2 < PieceStateNum; p2++) {
                for (int t = 0; t < ColorNum; t++) {
                    params.kpp[k1][p1][p2][t] -= (LearnEvalType)(LEARN_RATE * grad.kpp[k1][p1][p2][t]);
                }
            }
            for (int k2 = 0; k2 < SqNum; k2++) {
                for (int t = 0; t < ColorNum; t++) {
                    params.kkp[k1][k2][p1][t] -= (LearnEvalType)(LEARN_RATE * grad.kkp[k1][k2][p1][t]);
                }
            }
        }
    }
#endif
}

void BaseTrainer::updateParamsMomentum(EvalParams<LearnEvalType>& params, const EvalParams<LearnEvalType>& grad, EvalParams<LearnEvalType>& pre_update) {
#ifdef USE_NN
    for (int32_t i = 0; i < LAYER_NUM; i++) {
        auto curr_update_w = LEARN_RATE * grad.w[i] + MOMENTUM_DECAY * pre_update.w[i];
        auto curr_update_b = LEARN_RATE * grad.b[i] + MOMENTUM_DECAY * pre_update.b[i];
        params.w[i].array() -= curr_update_w.array();
        params.b[i].array() -= curr_update_b.array();
        pre_update.w[i] = curr_update_w;
        pre_update.b[i] = curr_update_b;
    }
#else
    for (int k1 = 0; k1 < SqNum; k1++) {
        for (int p1 = 0; p1 < PieceStateNum; p1++) {
            for (int p2 = 0; p2 < PieceStateNum; p2++) {
                for (int t = 0; t < ColorNum; t++) {
                    auto curr_update = (LearnEvalType)(LEARN_RATE * grad.kpp[k1][p1][p2][t] + MOMENTUM_DECAY * pre_update.kpp[k1][p1][p2][t]);
                    params.kpp[k1][p1][p2][t] -= curr_update;
                    pre_update.kpp[k1][p1][p2][t] = curr_update;
                }
            }
            for (int k2 = 0; k2 < SqNum; k2++) {
                for (int t = 0; t < ColorNum; t++) {
                    auto curr_update = (LearnEvalType)(LEARN_RATE * grad.kkp[k1][k2][p1][t] + MOMENTUM_DECAY * pre_update.kkp[k1][k2][p1][t]);
                    params.kkp[k1][k2][p1][t] -= curr_update;
                    pre_update.kkp[k1][k2][p1][t] = curr_update;
                }
            }
        }
    }
#endif
}

void BaseTrainer::timestamp() {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    auto minutes = seconds / 60;
    seconds %= 60;
    auto hours = minutes / 60;
    minutes %= 60;
    std::cout << std::setfill('0') << std::setw(3) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << "\t";
    log_file_ << std::setfill('0') << std::setw(3) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << "\t";
}

bool BaseTrainer::isLegalOptimizer() {
    return (OPTIMIZER_NAME == "SGD"
        || OPTIMIZER_NAME == "MOMENTUM");
}