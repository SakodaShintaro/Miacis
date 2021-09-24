#ifndef MIACIS_SHOGI_TEST_HPP
#define MIACIS_SHOGI_TEST_HPP

namespace Shogi {

void checkSearchSpeed();
void checkSearchSpeed2();
void checkGenSpeed();
void checkPredictSpeed();
void checkVal();
void checkValInfer();
void checkDoAndUndo();
void checkMirror();
void checkBook();
void makeBook();
void searchWithLog();
void testLoad();
void testModel();
void checkValLibTorchModel();
void checkLibTorchModel();
void checkLearningModel();
void checkInitLibTorchModel();
void checkValidData();
void checkBuildOnnx();

} // namespace Shogi

#endif //MIACIS_SHOGI_TEST_HPP