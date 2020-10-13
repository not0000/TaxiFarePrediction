using System;
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace TaxiFarePrediction
{
    /// <summary>
    /// 微軟範例試作 使用 ML.NET FastTree 決策樹回歸模型 預測計程車跳表價格
    /// https://docs.microsoft.com/zh-tw/dotnet/machine-learning/tutorials/predict-prices
    /// </summary>
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        
        /// <summary>
        /// 此問題是關於預測紐約市的計程車車資。 乍看之下，這可能看似單純取決於行程遠近。 
        /// 但紐約計程車廠商會針對其他因素 (例如額外的乘客，或使用信用卡而非現金付費) 收取不同的金額。
        /// 我們希望根據過往資料集的其他因素來預測價格值，這是一個實際值。 
        /// 為了執行該操作，您選擇迴歸機器學習服務工作。
        /// </summary>
        /// <param name="args"></param>

        static void Main(string[] args)
        {
            Console.WriteLine("從 csv 讀取1,048,575筆資料，使用 Fasttree 方式訓練模型");

            //讀取資料訓練模型 > 評估模型成效 > 跑預測結果
            Console.WriteLine("讀取csv資料訓練模型 > 評估模型成效 > 跑預測結果");
            TrainModel_Evaluate_Prediction();

            Console.WriteLine($"*************************************************"); 

            //讀取訓練完成的模型 >　跑預測結果
            Console.WriteLine("讀取已經訓練完成的模型 > 跑預測結果");
            LoadModel_Predciction();
        }

        private static void TrainModel_Evaluate_Prediction()
        {
            var StartTime = DateTime.UtcNow;
            var TimeCalc = "";
            TimeCalc += "Start" + (DateTime.UtcNow - StartTime) + Environment.NewLine;
            // 所有 ML.NET 作業都是從 MLContext 類別開始。 將 mlContext 初始化會建立新的 ML.NET 環境，可在模型建立工作流程物件間共用。 
            // 就概念而言，類似於 Entity Framework 中的 DBContext。
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);
            TimeCalc += "TrainFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;

            Evaluate(mlContext, model);
            TimeCalc += "EvaluateFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;

            TestSinglePrediction(mlContext, model);
            TimeCalc += "SinglePredictionFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;

            Console.WriteLine(TimeCalc);
        }

        private static void LoadModel_Predciction()
        {
            var StartTime = DateTime.UtcNow;
            var TimeCalc = "";
            TimeCalc += "Start" + (DateTime.UtcNow - StartTime) + Environment.NewLine;
            // 所有 ML.NET 作業都是從 MLContext 類別開始。 將 mlContext 初始化會建立新的 ML.NET 環境，可在模型建立工作流程物件間共用。 
            // 就概念而言，類似於 Entity Framework 中的 DBContext。
            MLContext mlContext = new MLContext(seed: 0);

            //var model = Train(mlContext, _trainDataPath);

            TimeCalc += "NoNeedTrainFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;

            // Load trained model
            DataViewSchema modelSchema;
            ITransformer trainedModel = mlContext.Model.Load("model.zip", out modelSchema);

            var model = trainedModel;

            var a = model.GetOutputSchema(modelSchema);

            TimeCalc += "LoadFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;



            //Evaluate(mlContext, model);
            TimeCalc += "NoNeedEvaluateFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;


            TestSinglePrediction(mlContext, model);
            TimeCalc += "SinglePredictionFinish" + (DateTime.UtcNow - StartTime) + Environment.NewLine;

            Console.WriteLine(TimeCalc);
        }


        /// <summary>
        /// 載入並轉換資料
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataPath"></param>
        /// <returns></returns>
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {

            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            // 使用 CopyColumnsEstimator 轉換類別來複製 FareAmount 
            // 練模型的演算法需要數值特徵，因此必須將類別資料（ VendorId 、 RateCode 和 PaymentType ）值轉換成數位 ( VendorIdEncoded 等三項)
            // 最後使用 mlContext.Transforms.Concatenate 轉換類別，將所有特徵資料行合併為 Features 資料行。 根據預設，學習演算法只會處理來自 Features 資料行的特徵。

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
            .Append(mlContext.Regression.Trainers.FastTree()); //將 FastTreeRegressionTrainer 機器學習服務工作新增到資料轉換定義

            var model = pipeline.Fit(dataView);

            SaveModel(mlContext, dataView, model);

            return model;
        }
        /// <summary>
        /// 儲存訓練完的模型
        /// https://docs.microsoft.com/zh-tw/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="dataView"></param>
        /// <param name="model"></param>
        private static void SaveModel(MLContext mlContext, IDataView dataView, TransformerChain<RegressionPredictionTransformer<FastTreeRegressionModelParameters>> model)
        {
            // 缺一段檢查是否有同名檔案，刪掉或是自動換檔名
            mlContext.Model.Save(model, dataView.Schema, "model.zip");
        }

        /// <summary>
        /// 使用測試資料評估模型效能，以進行品質保證和驗證。
        /// Evaluate 方法會執行下列工作：
        /// * 載入測試資料集。
        /// * 建立迴歸評估工具。
        /// * 評估模型並建立計量。
        /// * 顯示計量。
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // 使用 LoadFromTextFile() 方法載入測試資料集。
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            //針對測試資料集輸入資料列進行預測
            var predictions = model.Transform(dataView);

            // 使用指定的資料集來計算 PredictionModel 的品質計量。 傳回的 RegressionMetrics 物件包含迴歸評估工具所計算的整體計量
            RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       模型品質指標評估                         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        /// <summary>
        /// 使用模型來進行預測
        /// TestSinglePrediction 方法會執行下列工作：
        /// * 建立單一評論的測試資料。
        /// * 根據測試資料預測費用金額。
        /// * 合併測試資料和預測來進行報告。
        /// * 顯示預測的結果。
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            //測試用的計程車行程
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var taxiTripSample2 = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1260,
                TripDistance = 10.33f,
                PaymentType = "CSH",
                FareAmount = 0 // To predict. Actual/Observed = 29.5
            };
            var taxiTripSample3 = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 3,
                TripTime = 480,
                TripDistance = 1.9f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 8.5
            };


            var prediction = predictionFunction.Predict(taxiTripSample);
            var prediction2 = predictionFunction.Predict(taxiTripSample2);
            var prediction3 = predictionFunction.Predict(taxiTripSample3);

           

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"預測: {prediction.FareAmount:0.####}, 實際: 15.5, 誤差 " + Math.Round((Math.Abs(15.5 - prediction.FareAmount)) / 15.5,4)  +"%");
            Console.WriteLine($"預測: {prediction2.FareAmount:0.####}, 實際: 29.5, 誤差 " + Math.Round((Math.Abs(29.5 - prediction.FareAmount))/29.5,4) + "%");
            Console.WriteLine($"預測: {prediction3.FareAmount:0.####}, 實際: 8.5, 誤差 " + Math.Round(Math.Abs(8.5 - prediction.FareAmount)/8.5,4) + "%");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
