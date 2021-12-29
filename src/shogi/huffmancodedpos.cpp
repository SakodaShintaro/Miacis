#include "huffmancodedpos.hpp"

namespace Shogi {

// Binary表記
// Binary<11110>::value とすれば、30 となる。
// 符合なし64bitなので19桁まで表記可能。
template<uint64_t n> struct Binary { static const uint64_t value = n % 10 + (Binary<n / 10>::value << 1); };
// template 特殊化
template<> struct Binary<0> { static const uint64_t value = 0; };

// clang-format off
const HuffmanCode HuffmanCodedPos::boardCodeTable[PieceNum] = {
    {Binary<         0>::value, 1}, // Empty
    {Binary<         0>::value, 0}, // 1.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 2.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 3.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 4.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 5.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 6.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 7.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 8.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 9.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 10.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 11.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 12.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 13.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 14.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 15.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 16.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 17.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 18.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 19.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 20.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 21.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 22.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 23.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 24.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 25.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 26.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 27.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 28.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 29.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 30.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 31.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 32.使用しないので numOfBit を 0 にしておく。
    {Binary<         1>::value, 4}, // 33.BPawn
    {Binary<        11>::value, 6}, // 34.BLance
    {Binary<       111>::value, 6}, // 35.BKnight
    {Binary<      1011>::value, 6}, // 36.BSilver
    {Binary<      1111>::value, 6}, // 37.BGold
    {Binary<     11111>::value, 8}, // 38.BBishop
    {Binary<    111111>::value, 8}, // 39.BRook
    {Binary<         0>::value, 0}, // 40.BKing 玉の位置は別途、位置を符号化する。使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 41.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 42.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 43.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 44.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 45.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 46.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 47.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 48.使用しないので numOfBit を 0 にしておく。
    {Binary<      1001>::value, 4}, // 49.BProPawn
    {Binary<    100011>::value, 6}, // 50.BProLance
    {Binary<    100111>::value, 6}, // 51.BProKnight
    {Binary<    101011>::value, 6}, // 52.BProSilver
    {Binary<         0>::value, 0}, // 53.使用しないので numOfBit を 0 にしておく。
    {Binary<  10011111>::value, 8}, // 54.BHorse
    {Binary<  10111111>::value, 8}, // 55.BDragon
    {Binary<         0>::value, 0}, // 56.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 57.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 58.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 59.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 60.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 61.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 62.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 63.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 64.使用しないので numOfBit を 0 にしておく。
    {Binary<       101>::value, 4}, // 65.WPawn
    {Binary<     10011>::value, 6}, // 66.WLance
    {Binary<     10111>::value, 6}, // 67.WKnight
    {Binary<     11011>::value, 6}, // 68.WSilver
    {Binary<    101111>::value, 6}, // 69.WGold
    {Binary<   1011111>::value, 8}, // 70.WBishop
    {Binary<   1111111>::value, 8}, // 71.WRook
    {Binary<         0>::value, 0}, // 72.WKing 玉の位置は別途、位置を符号化する。
    {Binary<         0>::value, 0}, // 73.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 74.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 75.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 76.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 77.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 78.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 79.使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 80.使用しないので numOfBit を 0 にしておく。
    {Binary<      1101>::value, 4}, // 81.WProPawn
    {Binary<    110011>::value, 6}, // 82.WProLance
    {Binary<    110111>::value, 6}, // 83.WProKnight
    {Binary<    111011>::value, 6}, // 84.WProSilver
    {Binary<         0>::value, 0}, // 85.使用しないので numOfBit を 0 にしておく。
    {Binary<  11011111>::value, 8}, // 86.WHorse
    {Binary<  11111111>::value, 8}, // 87.WDragon
};

// 盤上の bit 数 - 1 で表現出来るようにする。持ち駒があると、盤上には Empty の 1 bit が増えるので、
// これで局面の bit 数が固定化される。
const HuffmanCode HuffmanCodedPos::handCodeTable[HAND_PIECE_KIND_NUM][ColorNum] = {
    {{Binary<        0>::value, 3}, {Binary<      100>::value, 3}}, // HPawn
    {{Binary<        1>::value, 5}, {Binary<    10001>::value, 5}}, // HLance
    {{Binary<       11>::value, 5}, {Binary<    10011>::value, 5}}, // HKnight
    {{Binary<      101>::value, 5}, {Binary<    10101>::value, 5}}, // HSilver
    {{Binary<      111>::value, 5}, {Binary<    10111>::value, 5}}, // HGold
    {{Binary<    11111>::value, 7}, {Binary<  1011111>::value, 7}}, // HBishop
    {{Binary<   111111>::value, 7}, {Binary<  1111111>::value, 7}}, // HRook
};
// clang-format on

HuffmanCodeToPieceHash HuffmanCodedPos::boardCodeToPieceHash;
HuffmanCodeToPieceHash HuffmanCodedPos::handCodeToPieceHash;
}