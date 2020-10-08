using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace TaxiFarePrediction
{
    /// <summary>
    /// TaxiTrip 是輸入資料類別，並含有每個資料集資料行的定義。
    /// 使用 [LoadColumnAttribute] 來指定資料集中來源資料行的索引。
    /// </summary>
    public class TaxiTrip
    {
        [LoadColumn(0)]
        public string VendorId;

        [LoadColumn(1)]
        public string RateCode;

        [LoadColumn(2)]
        public float PassengerCount;

        [LoadColumn(3)]
        public float TripTime;

        [LoadColumn(4)]
        public float TripDistance;

        [LoadColumn(5)]
        public string PaymentType;

        [LoadColumn(6)]
        public float FareAmount;
    }


    /// <summary>
    /// TaxiTripFarePrediction 類別代表預測的結果。 
    /// 它有一個已套用屬性的單一 float 欄位 FareAmount Score ColumnNameAttribute 。 
    /// 在回歸工作的情況下，[分數] 資料行會包含預測的標籤值。
    /// </summary>
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
